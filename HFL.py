
from options import args_parser
from tensorboardX import SummaryWriter
import torch
from client import Client
from cloud import Cloud
from edge import Edge
from datasets.get_data import get_dataloaders, show_distribution
import copy
import numpy as np
from tqdm import tqdm
from models.mnist_cnn import mnist_lenet
from models.cifar_cnn_3conv_layer import cifar_cnn_3conv
from models.cifar_resnet import ResNet18
from models.mnist_logistic import LogisticRegression
import math

from MAB import MAB
from PDP import PDP_select
import random

from model_collector import ModelCollector

import time 
t1 = 0
t2 = 0



def get_client_class(args, clients):
    client_class = []
    client_class_dis = [[],[],[],[],[],[],[],[],[],[]]
    for client in clients:
        train_loader = client.train_loader
        distribution = show_distribution(train_loader, args)
        label = np.argmax(distribution)
        client_class.append(label)
        client_class_dis[label].append(client.id)
    print(client_class_dis)
    return client_class_dis

def get_edge_class(args, edges, clients):
    edge_class = [[], [], [], [], []]
    for (i,edge) in enumerate(edges):
        for cid in edge.cids:
            client = clients[cid]
            train_loader = client.train_loader
            distribution = show_distribution(train_loader, args)
            label = np.argmax(distribution)
            edge_class[i].append(label)
    print(f'class distribution among edge {edge_class}')

def initialize_edges_iid(num_edges, clients, args, client_class_dis):
    """
    This function is specially designed for partiion for 10*L users, 1-class per user, but the distribution among edges is iid,
    10 clients per edge, each edge have 10 classes
    :param num_edges: L
    :param clients:
    :param args:
    :return:
    """
    #only assign first (num_edges - 1), neglect the last 1, choose the left
    edges = []
    p_clients = [0.0] * num_edges
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        for label in range(10):
        #     0-9 labels in total
            assigned_client_idx = np.random.choice(client_class_dis[label], 1, replace = False)
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
        # edges.append(Edge(id = eid,
        #                   cids=assigned_clients_idxes,
        #                   shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
        edges.append(Edge(id = eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers),
                          learning_rate=clients[0].model.learning_rate,
                          mab = MAB(args.mab_rate, int(args.num_clients/args.num_edges), clients),
                          args=args,
                          edge_clients = [i for i in clients if i.id in assigned_clients_idxes]
                          ))
        
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
        p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                        for sample in list(edges[eid].sample_registration.values())]
        edges[eid].refresh_edgeserver()
        
    #And the last one, eid == num_edges -1
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print("label{} is empty".format(label))
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    # edges.append(Edge(id=eid,
    #                   cids=assigned_clients_idxes,
    #                   shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
    edges.append(Edge(id = eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers),
                          learning_rate=clients[0].model.learning_rate,
                          mab = MAB(args.mab_rate, int(args.num_clients/args.num_edges), clients),
                          args=args
                          ))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
    p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                    for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients

def initialize_edges_niid(num_edges, clients, args, client_class_dis):
    """
    This function is specially designed for partiion for 10*L users, 1-class per user, but the distribution among edges is iid,
    10 clients per edge, each edge have 5 classes
    :param num_edges: L
    :param clients:
    :param args:
    :return:`
    """
    #only assign first (num_edges - 1), neglect the last 1, choose the left
    edges = []
    p_clients = [0.0] * num_edges
    # eid对应label范围
    label_ranges = [[0,1,2,3,4],[1,2,3,4,5],[5,6,7,8,9],[6,7,8,9,0]]
    
    # 创建数组存每个edge应该放多少个clients
    num_each_edge_list = [args.num_clients//args.num_edges] * args.num_edges
    # 最后一个 edge放多余的client, 即无法平分数量的 client
    num_each_edge_list[args.num_edges - 1] += (args.num_clients % args.num_edges) 
    
    
    # 对 num_edges - 1个 edge执行
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        # 用提前分组好的标签分布，配给当前edge
        label_range = label_ranges[eid]
        # 进行两次
        
        # 选出对应数量的client
        for i in range(num_each_edge_list[eid]):
            # 在eid对应label范围内随机选择label， 如果对应Label的客户端数组为空，则重新选
            # 一般情况下不会出现client不够用的情况，因此这里没有进行额外处理
            while (True):
                # 取余3防止超越index
                label = np.random.choice(label_ranges[eid % 4], 1, replace=False)[0]
                if len(client_class_dis[label]) > 0:
                    break
                else:
                    # 若不命中则从隔壁借
                    label = np.random.choice(label_ranges[(eid+1) % 4 ], 1, replace=False)[0]
                    if len(client_class_dis[label]) > 0:
                        break
                      
            # 对随机选取的标签对应的client数组中抽一个client_id
            assigned_client_idx = np.random.choice(client_class_dis[label], 1, replace=False)
            # 在字典中除去被抽取客户端的 id
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
            
            # 将抽取的客户端加入到注册列表
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
                    
        edges.append(Edge(id = eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers),
                          learning_rate=clients[0].model.learning_rate,
                          mab = MAB(args.mab_rate, int(args.num_clients/args.num_edges), clients),
                          args=args
                          ))
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
        p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                        for sample in list(edges[eid].sample_registration.values())]
        edges[eid].refresh_edgeserver()
        
    #And the last one, eid == num_edges -1
    #Find the last available labels
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print("label{} is empty".format(label))
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edges.append(Edge(id=eid,
                      cids=assigned_clients_idxes,
                      shared_layers=copy.deepcopy(clients[0].model.shared_layers),
                      learning_rate=clients[0].model.learning_rate,
                      args=args,
                      mab = MAB(args.mab_rate, int(args.num_clients/args.num_edges),clients)
                      ))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
    p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                    for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients

def all_clients_test(server, clients, cids, device):
    [server.send_to_client(clients[cid]) for cid in cids]
    for cid in cids:
        server.send_to_client(clients[cid])
        # The following sentence!
        clients[cid].sync_with_edgeserver()
    correct_edge = 0.0
    total_edge = 0.0
    for cid in cids:
        correct, total = clients[cid].test_model(device)
        correct_edge += correct
        total_edge += total
    return correct_edge, total_edge

def fast_all_clients_test(v_test_loader, global_nn, device):
    correct_all = 0.0
    total_all = 0.0
    with torch.no_grad():
        for data in v_test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = global_nn(inputs)
            _, predicts = torch.max(outputs, 1)
            total_all += labels.size(0)
            correct_all += (predicts == labels).sum().item()
    return correct_all, total_all

def initialize_global_nn(args):
    if args.dataset == 'mnist':
        if args.model == 'lenet':
            global_nn = mnist_lenet(input_channels=1, output_channels=10)
        elif args.model == 'logistic':
            global_nn = LogisticRegression(input_dim=1, output_dim=10)
        else: raise ValueError(f"Model{args.model} not implemented for mnist")
    elif args.dataset == 'cifar10' :
        if args.model == 'cnn_complex':
            global_nn = cifar_cnn_3conv(input_channels=3, output_channels=10)
        elif args.model == 'resnet18':
            global_nn = ResNet18()
        else: raise ValueError(f"Model{args.model} not implemented for cifar")
    elif args.dataset == 'cifar100':
        if args.model == 'resnet18':
            global_nn = ResNet18(num_classes=100)
        else: raise ValueError((f"Model{args.model} not implemented for cifar"))
    else: raise ValueError(f"Dataset {args.dataset} Not implemented")
    return global_nn

def create_clients_statues(num_clients, train_loaders, test_loaders, args, device, clients):

    min_idx_sample = min(len(loader.dataset) for loader in train_loaders)
    print("min is :",min_idx_sample)
    
    for i in range(num_clients):
        clients.append(
            Client(
                id=i,
                train_loader=train_loaders[i],
                test_loader=test_loaders[i],
                args=args,
                device=device,
                min_idx_sample=min_idx_sample,
            )
        )

def HFL(args):
    #make experiments repeatable
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')
    FILEOUT = f"_{args.remark}_policy{args.alg}_{args.dataset}_clients{args.num_clients}_edges{args.num_edges}_" \
              f"t1-{args.num_local_update}_t2-{args.num_edge_aggregation}" \
              f"_model_{args.model}iid{args.iid}edgeiid{args.edgeiid}epoch{args.num_communication}" \
              f"bs{args.batch_size}lr{args.lr}lr_decay_rate{args.lr_decay}" \
              f"lr_decay_epoch{args.lr_decay_epoch}momentum{args.momentum}using_dp{args.using_dp}"
    writer = SummaryWriter(comment=FILEOUT)
    # Build dataloaders
    train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataloaders(args)
    if args.show_dis:
        for i in range(args.num_clients):
            train_loader = train_loaders[i]
            print(len(train_loader.dataset))
            distribution = show_distribution(train_loader, args)
            print("train dataloader {} distribution".format(i))
            print(distribution)

        for i in range(args.num_clients):
            test_loader = test_loaders[i]
            test_size = len(test_loaders[i].dataset)
            print(len(test_loader.dataset))
            distribution = show_distribution(test_loader, args)
            print("test dataloader {} distribution".format(i))
            print(f"test dataloader size {test_size}")
            print(distribution)

    # initialize clients and server
    clients = []
    # clients状态初始化 + HierFL数据集初始化 + DP初始化
    create_clients_statues(args.num_clients, train_loaders, test_loaders, args, device, clients)

    initilize_parameters = list(clients[0].model.shared_layers.parameters())
    nc = len(initilize_parameters)
    for client in clients:
        user_parameters = list(client.model.shared_layers.parameters())
        
        # 
        
        for i in range(nc):
            user_parameters[i].data[:] = initilize_parameters[i].data[:]

    # Initialize edge server and assign clients to the edge server
    edges = []
    cids = np.arange(args.num_clients)
    clients_per_edge = int(args.num_clients / args.num_edges)
    p_clients = [0.0] * args.num_edges

    if args.iid == -2:
        if args.edgeiid == 1:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_iid(num_edges=args.num_edges,
                                                    clients=clients,
                                                    args=args,
                                                    client_class_dis=client_class_dis)
        elif args.edgeiid == 0:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_niid(num_edges=args.num_edges,
                                                     clients=clients,
                                                     args=args,
                                                     client_class_dis=client_class_dis)
    else:
        # This is randomly assign the clients to edges
        for i in range(args.num_edges):
            # 初始化edge
            #Randomly select clients and assign them
            selected_cids = np.random.choice(cids, clients_per_edge, replace=False)
            cids = list (set(cids) - set(selected_cids))
            edges.append(Edge(id = i,
                          cids=selected_cids,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers),
                          learning_rate=clients[0].model.learning_rate,
                          mab = MAB(args.mab_rate, int(args.num_clients/args.num_edges), clients),
                          args=args,
                          edge_clients = [i for i in clients if i.id in selected_cids]
                          ))
            # 这里注册Client，会把edge中的sample_registration字典给同步即cid:sample
            [edges[i].client_register(clients[cid]) for cid in selected_cids]

            edges[i].all_trainsample_num = sum(edges[i].sample_registration.values())
            # 从这里看，p大小取决于样本数量大小，样本多的客户端被选择的概率大
            p_clients[i] = [sample / float(edges[i].all_trainsample_num) for sample in
                    list(edges[i].sample_registration.values())]
            # 把edgeServer的receive_buffer、注册的用户id数组、sample_registration给清空
            edges[i].refresh_edgeserver()
            

    # Initialize cloud server
    cloud = Cloud(shared_layers=copy.deepcopy(clients[0].model.shared_layers))
    # First the clients report to the edge server their training samples
    [cloud.edge_register(edge=edge) for edge in edges]
    p_edge = [sample / sum(cloud.sample_registration.values()) for sample in
                list(cloud.sample_registration.values())]
    cloud.refresh_cloudserver()

    # New an NN model for testing error
    global_nn = initialize_global_nn(args)
    if args.cuda:
        global_nn = global_nn.cuda(device)

  
    # 画图数组
    all_counter = np.array([0]* args.num_communication * args.num_edge_aggregation)


    #!! 以下代码为收集模型信息并且构建模型反转攻击
    # 初始化追踪的客户端列表
    # model_collector = ModelCollector(args, clients)
    
    # # 根据是否使用DP选择追踪模式
    # if args.using_dp:
    #     tracked_clients = model_collector.initialize_dp_tracking()
    # else:
    #     random_tracked_client = model_collector.initialize_random_tracking()
    
    # Begin training
    for num_comm in tqdm(range(args.num_communication)):
        cloud.refresh_cloudserver()
        
        [cloud.edge_register(edge=edge) for edge in edges]
        for num_edgeagg in range(args.num_edge_aggregation):
            edge_loss = [0.0]* args.num_edges
            edge_loss_dp = [0.0]* args.num_edges
            edge_sample = [0]* args.num_edges
            # COCS 用
            edge_num_clients = [0]* args.num_edges            
            correct_all = 0.0
            total_all = 0.0

            # no edge selection included here
            # for each edge, iterate
            # COCS 画图计数器
            len_counter = 0
            for i,edge in enumerate(edges):
                
                edge.refresh_edgeserver()
                
                client_loss = 0.0
                client_loss_dp = 0.0
                selected_cnum = max(int(clients_per_edge * args.frac),1)
                # 存每一个edge所选择的客户端id
                selected_clients_id_list = []
                # 存每一个edge选择后， 能够如期到达的客户端的id
                reached_clients_id_list = []

                # ! ******************** Client Selection Policy *************************
                # * Randomly select 
                if args.alg == 'Random':
                    
                    t1 = time.time()
                    
                    edge_selected_clients = []
                    for client in clients:
                        if client.id in edge.cids:
                            edge_selected_clients.append(client)
                    
                    # random_rate = 0.2
                    selected_clients_list = random.sample(edge_selected_clients,int(args.random_rate*len(edge_selected_clients)))
                    selected_clients_id_list = [client.id for client in selected_clients_list] 
                    reached_clients_id_list = selected_clients_id_list
                    
                    
                    #更新选中客户端
                    selected_cids = selected_clients_id_list
                    [edge.client_register(clients[cid]) for cid in selected_cids]
                    
                    len_counter = len_counter + len(selected_clients_id_list)
                    
                    t2 = time.time()
                    print(f"time:{t2-t1},selected_clients_id_list:{selected_clients_id_list}")


                elif args.alg == 'Oort':
                    edge_selected_clients = []
                    edge_selected_clients_id = []
                    for client in clients:
                        if client.id in edge.cids:
                            edge_selected_clients.append(client)
                            edge_selected_clients_id.append(client.id)
                    
                    
                    
                    # 探索率和利用率计算
                    current_round = num_comm * args.num_edge_aggregation + num_edgeagg + 1
                    print("current_round {}".format(current_round))
                    
                    if current_round > 1 :
                        
                        exploration_factor_decay_rate = 0.98
                        exploration_factor_init = 0.9
                        exploration_factor = exploration_factor_init * (exploration_factor_decay_rate ** (current_round-1))
                    
                        # 探索比率
                        exploration_rate = args.mab_rate * exploration_factor
                        # 利用比率
                        exploitation_rate = 1 - exploration_factor
                    
                        # 获得的winner_list中，元素对应的client_id是由utility从大到小排序的
                        winner = edge.MAB.select(current_round,edge_selected_clients,args.mab_rate,args.remark)
                    
                        # 按利用比率选取
                        selected_clients_id_list = winner[:int(exploitation_rate * len(winner))]
                        # 剩下的名额由探索比例选取
                        remain_clients_id = list(set(edge_selected_clients_id) - set(selected_clients_id_list))
                        selected_clients_id_list.extend(random.sample(remain_clients_id, int(math.ceil(exploration_rate * len(remain_clients_id)))) )
                    else :
                        selected_clients_id_list = edge.MAB.select(current_round,edge_selected_clients,args.mab_rate,args.remark)
                    
                    # 使用选出的客户端的 id 制作新数组 A
                    reached_clients_id_list = selected_clients_id_list
                    
                    #更新选中客户端
                    selected_cids = reached_clients_id_list
                    [edge.client_register(clients[cid]) for cid in selected_cids]
                    
                    # 计数 
                    len_counter = len_counter + len(selected_clients_id_list)
                    
                    print("Oort selected:", reached_clients_id_list)
                
                elif args.alg == 'Poisson':

                    edge_selected_clients = []
                    for client in clients:
                        if client.id in edge.cids:
                            edge_selected_clients.append(client)
                    
                    # 生成泊松分布
                    lambda_param = args.Poisson_lambda  # 泊松分布的参数λ
                    
                    samples = np.random.poisson(lambda_param, int(len(edge_selected_clients)))
                    selected_indices = np.where(samples > 0)[0]
                    selected_clients = [edge_selected_clients[i] for i in selected_indices]
                    selected_clients_id_list = [i.id for i in selected_clients]
                    print("Poisson selected:", selected_clients_id_list)
                    for w in selected_clients :
                        print(f"Poisson winner: {w.id}, epsilon: {w.DP.dp_epsilon}, datasize: {len(w.train_loader)}, obf_utility: {w.obfuscated_training_utility}, prob: {w.probability}, utility: {w.mab_utility}, weight: {w.weight} ")
                    
                    selected_cids = selected_clients_id_list
                    [edge.client_register(clients[cid]) for cid in selected_cids]
                    
                    
                    
                elif args.alg == 'TPPS' or args.alg == 'TPPS2' :
                    edge_selected_clients = []
                    for client in clients:
                        if client.id in edge.cids:
                            edge_selected_clients.append(client)
                            
                    winner, _ = edge.TPPS.select(edge_clients=edge_selected_clients)
                    
                    selected_clients_id_list = [i.id for i in winner]
                    
                    # 打印winner信息
                    for w in winner :
                        print(f"TPPS winner: {w.id}, epsilon: {w.DP.dp_epsilon}, datasize: {len(w.train_loader)}, obf_utility: {w.obfuscated_training_utility}, prob: {w.probability}, utility: {w.mab_utility}, weight: {w.weight}, alpha: {edge.TPPS.alpha} ")
                    # print("TPPS selected:", selected_clients_id_list)
                    
                    # current_round从1开始
                    current_round = num_comm * args.num_edge_aggregation + num_edgeagg + 1
                    
                     
                    selected_cids = selected_clients_id_list
                    [edge.client_register(clients[cid]) for cid in selected_cids]
    
                
                # 发送边缘服务器的模型参数给选中的客户端
                if args.is_select_after_train == 0:
                    
                    t1 = time.time()
                    
                    for selected_cid in selected_cids:
                        edge.send_to_client(clients[selected_cid])
                        clients[selected_cid].sync_with_edgeserver()
                        if args.using_dp:
                            temp_client_loss, temp_client_loss_dp = clients[selected_cid].local_update(epoch=args.num_local_update,
                                                                        batch_size=args.batch_size,
                                                                        device = device,
                                                                        current_round = num_comm * args.num_edge_aggregation + num_edgeagg + 1)
                            client_loss += temp_client_loss
                            client_loss_dp += temp_client_loss_dp
                        else:
                            client_loss += clients[selected_cid].local_update(epoch=args.num_local_update,
                                                                            batch_size=args.batch_size,
                                                                        device = device,
                                                                        current_round = num_comm * args.num_edge_aggregation + num_edgeagg + 1)
                        clients[selected_cid].send_to_edgeserver(edge)
                    
                    t2 = time.time()
                    print("Learning Time: ", t2-t1)
                        
                    if args.using_dp:
                        edge_loss_dp[i] = client_loss_dp
                
                # 训练后进行客户端选择 模式：select_after_train
                elif args.is_select_after_train == 1 and args.alg  == 'PDP':
                    
                    selected_cids = []
                    loss_dict = {}
                    loss_dp_dict = {}
                
                    edge_selected_clients = []
                    for client in clients:
                        if client.id in edge.cids:
                            edge_selected_clients.append(client)
                    
                    # 把模型发给所有客户端
                    for client in edge_selected_clients:
                        # edge把模型发到edge范围内的客户端的buffer
                        edge.send_to_client(client)
                        # 客户端从buffer接受模型
                        client.sync_with_edgeserver()
                        
                    
                    if args.using_dp:
                            current_round = num_comm * args.num_edge_aggregation + num_edgeagg + 1 
                            total_round = args.num_communication * args.num_edge_aggregation + num_edgeagg + 1 
                            for client in edge_selected_clients:
                                temp_client_loss, temp_client_loss_dp = client.local_update(epoch=args.num_local_update,
                                                                            batch_size=args.batch_size,
                                                                            device = device,
                                                                            current_round=current_round)
                                # 记录loss && loss_dp
                                client_loss += temp_client_loss
                                client_loss_dp += temp_client_loss_dp
                                loss_dict[client.id] = temp_client_loss
                                loss_dp_dict[client.id] = temp_client_loss_dp
                                
                            # 选择出winner组客户端
                            selected_cids = PDP_select(edge_selected_clients, loss_dict, args.pdp_rate, current_round, total_round)
                            
                            print("selected_clients_id_list:",selected_cids)
                            # 更新选中客户端
                            [edge.client_register(clients[cid]) for cid in selected_cids]
                            # 发送模型给edge
                            for selected_cid in selected_cids:
                                clients[selected_cid].send_to_edgeserver(edge)
                            # 统计边缘loss
                            edge_loss_dp[i] = client_loss_dp
                            
                            selected_clients_id_list = selected_cids 

                    else:
                            print("PDP requires using dp !!!")
                            return 0
                
                
                edge_loss[i] = client_loss
                edge_sample[i] = sum(edge.sample_registration.values())
                edge_num_clients[i] = len(selected_clients_id_list)

                edge.aggregate(args)
         
                correct, total = all_clients_test(edge, clients, selected_cids, device)
                correct_all += correct
                total_all += total
            
            # MAB loss with DP
            all_loss = sum([e_loss * e_sample for e_loss, e_sample in zip(edge_loss, edge_sample)]) / sum(edge_sample)
            all_loss = 0
            all_loss_dp = 0
            for i in range(len(edge_loss)):
                avg_edge_i_loss = edge_loss[i] / edge_num_clients[i]
                all_loss += avg_edge_i_loss
                if args.using_dp:
                    avg_edge_i_loss_dp = edge_loss_dp[i] / edge_num_clients[i]
                    all_loss_dp += avg_edge_i_loss_dp
                    
            # DP loss    
            all_loss = all_loss / args.num_edges
            if args.using_dp:
                all_loss_dp = all_loss_dp / args.num_edges
                writer.add_scalar(f'Partial_Avg_Train_loss_dp',
                                all_loss_dp,
                                num_comm * args.num_edge_aggregation + num_edgeagg + 1 )
                
                writer.add_scalar(f'Difference_LossDP_Loss',
                                  all_loss_dp - all_loss,
                                  num_comm* args.num_edge_aggregation + num_edgeagg + 1
                                  )
            
            avg_acc = correct_all / total_all
            
            
            writer.add_scalar(f'Partial_Avg_Train_loss',
                          all_loss,
                          num_comm* args.num_edge_aggregation + num_edgeagg + 1)
            writer.add_scalar(f'All_Avg_Test_Acc_edgeagg',
                          avg_acc,
                          num_comm * args.num_edge_aggregation + num_edgeagg + 1)
            
            # 
            all_counter[num_comm * args.num_edge_aggregation + num_edgeagg] = len_counter
            
        for edge in edges:
            edge.send_to_cloudserver(cloud)
        cloud.aggregate(args)
        for edge in edges:
            cloud.send_to_edge(edge)

        global_nn.load_state_dict(state_dict = copy.deepcopy(cloud.shared_state_dict))
        global_nn.train(False)
        correct_all_v, total_all_v = fast_all_clients_test(v_test_loader, global_nn, device)
        avg_acc_v = correct_all_v / total_all_v
        writer.add_scalar(f'All_Avg_Test_Acc_cloudagg_Vtest',
                          avg_acc_v,
                          num_comm + 1)

    writer.close()
    print(f"The final virtual acc is {avg_acc_v}")


def main():
    args = args_parser()
    HFL(args)

if __name__ == '__main__':
    main()
