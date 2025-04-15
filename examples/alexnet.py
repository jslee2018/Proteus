import os
import json
import argparse
import proteus
import proteus.torchapi as torch
import proteus.torchapi.nn as nn
from proteus.simulator.simulator import Simulator

# from proteus.torchapi.nn.cube import DeviceCube
# from proteus.ir import ProteusModel, graph
# from proteus.algorithm.dm_algo import DMAlgo
# from proteus.algorithm.ilp_algo import ILPAlgo
# from proteus.algorithm.ilp_stage import StageILP

from models import AlexNet, inception_v3, resnet50, resnet18, vgg19
from build_dev_topo import build_cluster

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='alexnet')
parser.add_argument('-bs', type=int, default=128)
parser.add_argument('-ps', type=str, default='manual')
parser.add_argument('-ndev', type=int, default=4)
parser.add_argument('-cluster', type=str, default='n1_g1')
parser.add_argument('--disable-collective', action='store_true')
parser.add_argument('--bucket-size', type=int, default=25)
parser.add_argument('--reprofile', action='store_true')
parser.add_argument('--profile-iters', type=int, default=10)
parser.add_argument('--test', action='store_true')
parser.add_argument('--flexflow', action='store_true')
parser.add_argument('--tp-degree', type=int, default=1)
parser.add_argument('--pp-degree', type=int, default=2)
parser.add_argument('--dp-degree', type=int, default=2)
args = parser.parse_args()

def split_pp(stree, pp_degree, dev_topo, ndev, stage_map=None):
    """
    Pipeline Parallelism 균등 분할 함수.
    
    인자:
      stree      : 전체 Strategy Tree
      pp_degree  : Pipeline stage 수 (현재 2만 지원)
      dev_topo   : 디바이스 토폴로지 객체 (Proteus 제공)
      ndev       : 전체 사용 디바이스 수
      stage_map  : 각 stage에 할당할 모듈 이름 리스트를 담은 dict
                   기본값 (AlexNet 예시):
                     {0: ['features'], 1: ['avgpool', 'seq2', 'classifier', 'criterion']}
                     
    반환:
      각 stage에 해당하는 디바이스 mesh 리스트 (예: [stg0, stg1])
    """
    
    # 기본 stage_map 설정 (필요시 변경)
    if stage_map is None:
        stage_map = {
            0: ['features'],
            1: ['avgpool', 'seq2', 'classifier', 'criterion']
        }
        
    devs = list(range(ndev))
    num_per_stage = ndev // pp_degree
    stage_meshes = []
    
    for stage in range(pp_degree):
        stage_devs = devs[stage * num_per_stage : (stage + 1) * num_per_stage]
        mesh = dev_topo.make_mesh(stage_devs)
        stage_meshes.append(mesh)
        
        # 이 stage에 할당된 모듈들에 대해 split & map 실행
        for module_name in stage_map.get(stage, []):
            module = getattr(stree, module_name, None)
            if module is not None:
                module.split(0, len(stage_devs))
                module.map(mesh)
            else:
                print("Warning: stree에 {} 모듈이 없습니다.".format(module_name))
    return stage_meshes

def split_tp(module, tp_degree, mesh_tp):
    """
    지정된 모듈(module)에 대해 Tensor Parallelism 적용.
    - tp_degree: 텐서를 균등하게 분할할 개수
    - mesh_tp: 해당 TP group에 매핑할 디바이스 mesh (예: (dp_degree, tp_degree))
    여기서는 예시로 AlexNet의 classifier 내부 서브모듈에 대해
    미리 정의한 split 차원(dict)을 사용합니다.
    """
    # 예시 split 차원 맵 (AlexNet 기반)
    split_dims = {
        'seq0': 0,
        'seq1': 1,
        'seq2': 1,
        'seq3': 1,
        'seq4': 2,
        'seq5': 0,
        'seq6': 0,
    }
    for name, dim in split_dims.items():
        if hasattr(module, name):
            node = getattr(module, name)
            node.split(dim, tp_degree)
            node.map(mesh_tp)

def split_dp(stage, dp_degree, mesh_dp):
    """
    Data Parallelism 적용 함수.
    stage: 해당 Strategy Tree (또는 서브 모듈)
    dp_degree: Data Parallel degree (입력 배치 split)
    mesh_dp: DP에 해당하는 1D 디바이스 mesh (예: (dp_degree,))
    """
    stage.split(0, dp_degree)
    stage.map(mesh_dp)
    if hasattr(stage, 'optimizer'):
        stage.optimizer.split(0, 1)
        stage.optimizer.map(mesh_dp)

if __name__ == '__main__':
    # algorithm
    if args.model.lower() == 'alexnet':
        model = AlexNet()
        img_shape = (3, 224, 224)
    elif args.model.lower() == 'inception_v3':
        model = inception_v3(aux_logits=False)
        img_shape = (3, 224, 224)
    elif args.model.lower() == 'resnet50':
        model = resnet50()
        img_shape = (3, 224, 224)
    elif args.model.lower() == 'resnet18':
        model = resnet18()
        img_shape = (3, 224, 224)
    elif args.model.lower() == 'vgg19':
        model = vgg19()
        img_shape = (3, 224, 224)
    else:
        print('Unknown model: {}'.format(args.model))
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # build device topo cluster
    with open(args.cluster, 'r') as f:
        cluster_info = json.load(f)
    cluster = build_cluster(topo_file='{}/topos/topo-n{}.xml'.format(
        os.path.dirname(args.cluster), cluster_info['n_gpu_per_node']),
                            **cluster_info)

    bs = args.bs * cluster.n_node * cluster.n_gpu_per_node
    inputs = {
        'input': (tuple([bs] + list(img_shape)), ),  # tuple of input shape
        'label': ((bs, ), )  # tuple of label shape
    }
    graph, stree = proteus.compile(model, inputs, criterion, optimizer)

    args.ndev = cluster.n_node * cluster.n_gpu_per_node

    stree.init_config(cluster.dev_topo, stride=2)
    dev_topo = stree.dev_topo
    dp_mesh = dev_topo.create_mesh((args.ndev, ))
    if args.ps == 'manual':
        stree.root.split(0, args.ndev)
        stree.root.map(dp_mesh)

        if args.model.lower() == 'alexnet':
            stree.classifier.seq0.split(0, 1, item='out')
            stree.classifier.seq0.map(dp_mesh, item='out')
            stree.classifier.seq0.split(0, args.ndev, item='out_grad')
            stree.classifier.seq0.map(dp_mesh, item='out_grad')
            stree.classifier.seq1.split(1, args.ndev)
            stree.classifier.seq2.split(1, args.ndev)
            stree.classifier.seq3.split(1, args.ndev)
            stree.classifier.seq4.split(2, args.ndev)
            stree.classifier.seq4.split(0, args.ndev, item='out')
            stree.classifier.seq4.map(dp_mesh, item='out')
            stree.classifier.seq5.split(0, args.ndev)
            stree.classifier.seq6.split(0, args.ndev)
        elif args.model.lower() in ['resnet50', 'inception_v3']:
            stree.fc.split(1, args.ndev)
            stree.fc.map(dp_mesh)
            stree.fc.split([0, 1], [1, 1], item='out')
            stree.fc.map(dp_mesh, item='out')
        elif args.model.lower() == 'vgg19':
            stree.seq2.split(0, 1, item='out')
            stree.seq2.map(dp_mesh, item='out')
            stree.seq2.split(0, args.ndev, item='out_grad')
            stree.seq2.map(dp_mesh, item='out_grad')
            stree.classifier.seq0.split(1, args.ndev)
            stree.classifier.seq1.split(1, args.ndev)
            stree.classifier.seq2.split(1, args.ndev)
            stree.classifier.seq3.split(2, args.ndev)
            stree.classifier.seq3.split(0, args.ndev, item='out')
            stree.classifier.seq3.map(dp_mesh, item='out')

            stree.classifier.seq4.split(0, args.ndev)
            stree.classifier.seq5.split(0, args.ndev)
            stree.classifier.seq6.split(0, args.ndev)

            # stree.classifier.seq5.split(0, 1, item='out')
            # stree.classifier.seq5.map(dp_mesh, item='out')
            # stree.classifier.seq5.split(0, args.ndev, item='out_grad')
            # stree.classifier.seq5.map(dp_mesh, item='out_grad')

            # stree.classifier.seq6.split(1, args.ndev)
            # stree.classifier.seq6.map(dp_mesh)
            # stree.classifier.seq6.split([0, 1], [1, 1], item='out')
            # stree.classifier.seq6.map(dp_mesh, item='out')

        stree.schedule()
    elif args.ps == 'dp':
        stree.root.split(0, args.ndev)
        stree.root.map(dp_mesh)

        # offload example
        # cpu_mesh = dev_topo.create_mesh((args.ndev, ), type='cpu')
        # stree.classifier.split(0, args.ndev, item='weight')
        # stree.classifier.map(cpu_mesh, item='weight')

        stree.optimizer.split(0, 1)
        stree.optimizer.map(dp_mesh)
        stree.schedule()
    elif args.ps == 'pp':
        devs = [i for i in range(args.ndev)]
        stg_1 = dev_topo.make_mesh(devs[:args.ndev//2])
        stree.features.split(0, len(stg_1))
        stree.features.map(stg_1)

        stg_2 = dev_topo.make_mesh(devs[args.ndev//2:])
        stree.avgpool.split(0, len(stg_2))
        stree.seq2.split(0, len(stg_2))
        stree.classifier.split(0, len(stg_2))
        stree.criterion.split(0, len(stg_2))
        stree.avgpool.map(stg_2)
        stree.seq2.map(stg_2)
        stree.classifier.map(stg_2)
        stree.criterion.map(stg_2)

        stree.schedule(n_macro_batch=2,
                       interleave_freq=1,
                       max_ongoing_macro_batch=2)
    elif args.ps == '3d':
        # 사용자로부터 각 parallelism degree를 입력받음
        dp_degree = args.dp_degree       # Data Parallel degree
        pp_degree = args.pp_degree       # Pipeline Parallel degree (현재 2 지원)
        tp_degree = args.tp_degree       # Tensor Parallel degree

        # 전체 디바이스 수가 dp * pp * tp와 일치하는지 확인
        if dp_degree * pp_degree * tp_degree != args.ndev:
            print("Error: dp_degree * pp_degree * tp_degree must equal ndev!")
            exit(1)

        # 전체 3D mesh 생성: shape = (dp_degree, pp_degree, tp_degree)
        full_mesh = dev_topo.create_mesh((dp_degree, pp_degree, tp_degree))
        
        # 1) Pipeline Parallelism 분할: split_pp()를 이용하여 전체 Strategy Tree를 pp_degree stage로 균등 분할.
        #    이때 각 stage에 대해 stage_map에 따라 일부 모듈(split, map)이 이미 처리됨.
        pp_meshes = split_pp(stree, pp_degree, dev_topo, args.ndev)
        
        # 2) 각 pipeline stage 내에서 추가적으로 Tensor Parallelism 적용.
        #    예시로, AlexNet의 경우 stage 1 (후반부)에 해당하는 'classifier' 모듈에 대해 TP 적용
        #    full_mesh[:, 1, :] 는 pp 축 인덱스 1에 해당하는 (dp_degree x tp_degree) mesh를 의미.
        if args.model.lower() == 'alexnet':
            tp_mesh = full_mesh[:, 1, :]
            split_tp(stree.classifier, tp_degree, tp_mesh)
        
        # 3) Data Parallelism 적용: 전체 모델의 입력 부분에 대해 DP split을 적용.
        stree.root.split(0, dp_degree)
        dp_mesh = full_mesh[:, 0, 0]
        stree.root.map(dp_mesh)

        # 4) 최종 스케줄링: 보통 n_macro_batch를 dp_degree로 설정하여 DP 그룹 단위의 batch 처리를 모사
        stree.schedule(n_macro_batch=dp_degree,
                       interleave_freq=1,
                       max_ongoing_macro_batch=2)

    if args.disable_collective:
        stree.disable_collective_comm()
    stree.set_bucket_size(args.bucket_size)

    graph.init_config(stree)
    stree.propagate(graph)
    graph.propagate({})
    graph.symmetric_forward_backward()

    # graph.to_graphviz()

    graph.export_config('config.txt')

    stree.dump_tree(config=False)

    if 'titan' in args.cluster or '1080' in args.cluster:
        overlap_factor = 0.3
    else:
        overlap_factor = 0.3
    if args.flexflow:
        overlap_factor = 0
    sim = Simulator(graph,
                    stree,
                    cost_type='profile',
                    reprofile=args.reprofile,
                    profile_iters=args.profile_iters,
                    optimizer_overlap=False,
                    cprofile_compile=False,
                    share_bandwidth=(not args.flexflow),
                    overlap_factor=overlap_factor,
                    FlexFlow=args.flexflow)
    cost = sim.run('log/trace', cprofile_analysis=False)

    sim.print_stats()
