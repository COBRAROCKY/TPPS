

import numpy as np
import math 



class MAB():
    def __init__(self, mab_rate, num_client, clients):
        self.mab_rate = mab_rate
        self.num_client = num_client
        # 所有联邦学习的clients 
        self.clients = clients
        self.current_round = 0
        
        
        self.util_list = []
        self.util_dict_list = []
        
        # 初始化选择
        self.client_selected_cnt = {client.id:0 for client in clients}
            
    
    def select(self, current_round, edge_clients, rate , remark):
        
         
        mab_utility_list = []
        e_dict = {}
        e_list = []
        f_list = []
        f_dict = {}
        num_select = self.num_client * rate
        
        # * 计算 MAB Utility
        for client in edge_clients:
            mab_utility_list.append(client.mab_utility)
            

        
        # * 计算 e-index 相关
        # * the added item is \sqrt{(K+1)*ln(t)/\gamma_{i,t}}
        if (current_round > 1):
            for client in edge_clients:
                
                # 旧

                e_index = math.sqrt(
                    math.log(current_round) / (num_select * self.client_selected_cnt[client.id])
                )
                
                f_index = math.sqrt(
                    num_select * self.client_selected_cnt[client.id] / math.log(current_round)
                ) * client.DP.dp_epsilon / client.DP.sensitivity
                
                e_dict[client.id] = e_index
                e_list.append(e_index)
                f_dict[client.id] = f_index
                f_list.append(f_index)
        
        
        utilities = []
        winner = []

        if current_round == 1:
            winner = [client.id for client in edge_clients]
        else :
            for client in edge_clients:
                index = client.id 
                utility = client.mab_utility
                
                # CIFAR-10用
                beta = 1
                alpha = len(client.train_loader.dataset)

                if ("Oort" in remark):
                    utility = client.mab_utility
                
                utilities.append([index,utility])
                # print(f"exp_id : {remark} idx : {index} , e_index : {e_dict[index] * alpha} , f_index : {f_dict[index]* beta} , utility : {client.mab_utility} , sensitivity : {client.DP.sensitivity} , epsilon : {client.DP.dp_epsilon} , total : {utility} , selected_cnt: {self.client_selected_cnt[client.id]}")
                
                
            # 选择Top K个客户端 utilities 从大到小排序
            utilities.sort(key=lambda x:x[1], reverse=True)
            # utilities.sort(key=lambda x:x[1], reverse=False)
            # 选择utilities大的K个
            winner = [utility[0] for utility in utilities[:int(self.num_client * self.mab_rate)]]


        for i in winner:
            self.client_selected_cnt[i] += 1
            
        return winner
