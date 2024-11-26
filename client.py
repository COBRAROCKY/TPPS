# The structure of the client
# Should include following funcitons
# 1. Client intialization, dataloaders, model(include optimizer)
# 2. Client model update
# 3. Client send updates to server
# 4. Client receives updates from server
# 5. Client modify local model based on the feedback from the server
from torch.autograd import Variable
import torch
from models.initialize_model import initialize_model
import copy
import numpy as np
from dp.DP import DP
import math
import random
import time 

class Client():

    def __init__(self, id, train_loader, test_loader, args, device, min_idx_sample, Ynt=0, dist=0, charge=0, bandwidth=0):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        # copy.deepcopy(self.model.shared_layers.state_dict())
        self.receiver_buffer = {}
        self.batch_size = args.batch_size
        #record local update epoch
        self.epoch = 0
        # record the time
        self.clock = []
        self.min_idx_sample = min_idx_sample


        
        # MAB用 上一轮的平均loss,utility
        self.loss_last_round = 0
        self.mab_utility = 0
        self.sensitivity = 0
        
        
        #TPPS用
        self.alg = args.alg
        self.weight = 1
        self.probability = 0
        self.obfuscated_training_utility = 0
        # self.utility_sensitivity = np.sqrt( 5000 * (5000 * (23026 **2))) 
        self.utility_sensitivity = 1
        # self.utility_max = 11513.0
        self.client_selected_cnt = 0

        if args.alg == 'TPPS':
            self.utility_max = 6000.0
            self.TPPS_rate = args.TPPS_rate
            self.num_edge_clients = int(args.num_clients / args.num_edges)
            
        self.noise_rate = np.clip(self.noise_rate, 0.1, 0.9)

        # DP用
        self.using_dp = args.using_dp
        self.loss_dp_eval = args.loss_dp_eval
        if self.using_dp:
            self.DP = DP(   dp_mechanism=args.dp_mechanism if args.dp_mechanism is not None else None,
                            times=args.num_local_update * args.num_edge_aggregation,
                            # dp_epsilon=args.dp_epsilon if args.dp_epsilon is not None else None,
                            dp_epsilon=random.randint(2, 10),
                            # dp_epsilon=args.dp_epsilon,
                            dp_delta=args.dp_delta if args.dp_delta is not None else None,
                            # idxs=args.batch_size * args.num_local_update,
                            min_idx_sample = min_idx_sample,
                            learning_rate=self.model.learning_rate,
                            dp_clip=args.dp_clip if args.dp_clip is not None else None,
                            using_dp=args.using_dp
                            
                         )
        
        # 模拟clients的数据标签错误现象
        if args.is_noise_labels : 
            client_data_idx = np.array(self.train_loader.dataset.idxs)
            original_dataset = self.train_loader.dataset.dataset 
            # 根据数据集取得源头所有label
            if args.dataset != 'mnist':
                original_labels = np.array(self.train_loader.dataset.dataset.targets)
            else :
                original_labels = np.array(self.train_loader.dataset.dataset.labels)
            # 取得对应的label的array
            client_labels = original_labels[client_data_idx]
            # 取得需要更改的标签的索引(关于client_labels)
            indices_to_change = random.sample(range(len(client_labels)),  int(len(client_labels) * self.noise_rate))
            for idx in indices_to_change:
                original_label = client_labels[idx]
                # 确保修改后的标签与原标签不同
                wrong_label = original_label
                while wrong_label == original_label:
                    if args.dataset == 'cifar100':
                        wrong_label = np.random.randint(0,100)
                    else:
                        wrong_label = random.randint(0, 9)
                client_labels[idx] = wrong_label
            for idx, new_label in zip(client_data_idx, client_labels):
                original_dataset.targets[idx] = new_label 


    def local_update(self, epoch, batch_size, device, current_round):
        # iter_now 为 num_local_update
        iter_now = 0
        loss = 0.0
        loss_sqrt = 0.0
        end = False

        for e in range(epoch):
            for data in self.train_loader:
                
                inputs, labels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                temp_loss_list = self.model.optimize_model(input_batch=inputs,
                                                    label_batch=labels,
                                                    obj=self) 
                if (e == 0):
                    # iter_now计数器
                    iter_now += 1 
                    loss_list =  temp_loss_list.cpu().detach().numpy()
                    # temp_loss为平均损失
                    for item in loss_list:
                        loss_sqrt += (item **2) 
                        loss += item
           
            self.epoch += 1
            self.model.exp_lr_sheduler(epoch = self.epoch)    
    
        loss /= (iter_now * batch_size)  
        
        # 完成n个epoch后进行梯度裁剪和模型加噪
        # 梯度裁剪   ****************
        # start = time.time()
        if self.using_dp:
            # print("check for clipping")
            self.DP.clip_gradients(model=self.model.shared_layers)
            
        # 模型加噪 ******************
            # print("check for adding noise")
            self.DP.add_noise(model=self.model.shared_layers)
            # end_t = time.time()
            # print("加噪用时：",end - start)
            
        #  补充加噪后的loss计算，评估模式***
            # start_t = time.time()
            if self.loss_dp_eval:
                loss_dp = 0
                # 评估模式
                self.model.shared_layers.eval()
                temp_num_iter = 0
                for data in self.test_loader:
                    inputs, labels = data
                    inputs = Variable(inputs).to(device)
                    labels = Variable(labels).to(device)
                    with torch.no_grad():
                        outputs = self.model.shared_layers(inputs)
                        loss_dp += self.model.criterion(outputs, labels).item()** 2
                        # loss_dp += self.model.criterion(outputs, labels).item() 
                    temp_num_iter += 1
                    
                # end_t = time.time()
                # print("测试加噪模型用时:",end-start)
                loss_dp /= temp_num_iter
                
                self.loss_last_round = loss
                # self.mab_utility = math.sqrt(loss_sqrt / (num_iter*batch_size) ) * (num_iter*batch_size)
                self.mab_utility = math.sqrt(loss_sqrt / (len(self.train_loader.dataset) )) * (len(self.train_loader.dataset))
                
                return loss, loss_dp
     
            self.loss_last_round = loss
            # self.mab_utility = math.sqrt(loss / (num_iter*batch_size)) * (num_iter*batch_size)
            self.mab_utility = math.sqrt(loss_sqrt / (len(self.train_loader.dataset) )) * (len(self.train_loader.dataset))
            
            # 此处client先使用理论utility最大值进行归一化，然后对其加噪后上传
            if self.alg == 'TPPS':
                if self.client_selected_cnt == 0:
                    self.obfuscated_training_utility = self.DP.add_noise_Lap(parameter=min(1, (self.mab_utility/self.utility_max)), sensitivity=self.utility_sensitivity, budget=self.DP.dp_epsilon)
                else:
                    beta = 1
                    alpha = len(self.train_loader.dataset)
                    e_index = math.sqrt(
                    (math.log(current_round)+0.1) / (self.num_edge_clients * self.client_selected_cnt)
                    )
                    f_index = math.sqrt(
                        self.num_edge_clients  * self.client_selected_cnt / (math.log(current_round)+0.1)
                    ) * self.DP.dp_epsilon / self.DP.sensitivity
                    utility = self.mab_utility + e_index * alpha + f_index * beta
                   
                    self.mab_utility = utility
                    # 5：让其映射到0-10区间，增大差距
                    expanded_variance = 10
                    self.obfuscated_training_utility, noise  =  self.DP.add_noise_Lap(parameter=( utility/self.utility_max), sensitivity=self.utility_sensitivity, budget=self.DP.dp_epsilon)
                    self.obfuscated_training_utility = self.obfuscated_training_utility * expanded_variance
                    print(f"current_round: {current_round}, id : {self.id}, total_utility : {self.mab_utility}, e_index : {e_index*alpha}, f_index : {f_index*beta}, obf_utility : {self.obfuscated_training_utility}, noise : {noise}, selected_cnt: {self.client_selected_cnt}")

            # return loss , loss_dp
            return loss , 0  
        
        # * 无加噪，MAB用 self为 client
        self.loss_last_round = loss
        self.mab_utility = math.sqrt(loss_sqrt / (len(self.train_loader.dataset) )) * (len(self.train_loader.dataset))
           
        return loss

    def test_model(self, device):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch= inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(client_id= self.id,
                                        cshared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
                                        )
        return None

    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict
        return None

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # self.model.shared_layers.load_state_dict(self.receiver_buffer)
        self.model.update_model(self.receiver_buffer)
        return None
