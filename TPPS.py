import numpy as np 
import random 
import math 
import copy

class TPPS():
    def __init__(self, rate, total_clients, total_rounds, ):
        self.current_round = 0
        
        # B = 5   # of picked workers
        # count = 20
        self.alpha_step=10
        self.k = rate * len(total_clients)
        self.miu = 1
        self.delta_u = 1
        self.N = len(total_clients)
        
        # 记录使用
        self.alpha = 0
        
        self.gamma = np.sqrt(
            (
                (1 - self.k / (2 * self.N)) * self.miu * np.sqrt(sum([(self.delta_u ** 2) / (i.DP.dp_epsilon ** 2) for i in total_clients])) + self.N
            ) / (
                    (np.exp(1) - 1 ) * total_rounds * self.k
                ) 
                * np.log(self.N/self.k)
            )



    def select(self, edge_clients):
        winner, total= self.TPPS_selection_policy(temp_worker_list=edge_clients, k=self.k)
        for w in winner :
            w.client_selected_cnt += 1
            
        return winner, total
                

    def TPPS_selection_policy(self, temp_worker_list, k):
        
            N = len(temp_worker_list)
            delta = (1 / k - self.gamma / N) / (1 - self.gamma)
            weight_list = [i.weight for i in temp_worker_list]
            
            # 找阈值
            if np.max(weight_list)/np.sum(weight_list) > delta:
                temp_worker_list = sorted(temp_worker_list, key=lambda x: x.weight, reverse=True)
                weight_list = [i.weight for i in temp_worker_list]
                # Alpha文中为Lambda
                alpha, error = self.findAlpha(temp_list=weight_list, delta=delta)
                for w in temp_worker_list:
                    if w.weight > alpha:
                        w.weight = alpha
            else:
                alpha = np.max(weight_list) + 1
            sum_weight = np.sum([i.weight for i in temp_worker_list])
            
            # 计算每个客户端的概率
            # j = 0
            for w in temp_worker_list:
                w.probability = k * ((1 - self.gamma) * w.weight / sum_weight + self.gamma / N)
                # j = j + 1
                # w.probability_list.append(w.probability)
                # w.probability_2 = w.probability
               
            # 找出t时刻的winner , winner为client对象list
            temp_winner_list = self.DependentRound(worker_list=temp_worker_list, k=k)
            
            
            if(len(temp_winner_list) != k):
                print(f'Error of DependentRound: k:{k}, size of winners:{len(temp_winner_list)}')
            
            
            # TODO 更新每个客户端的权重 (根据选择结果来更新)
            # obfuscated_training_utility 已经为归一化结果(客户端上传时)
            
            for winner in temp_winner_list:
                # obfuscated utility 
                temp_utility = winner.obfuscated_training_utility  
                if winner.weight < alpha :
                    temp_probability = k * ((1 - self.gamma) * winner.weight / sum_weight + self.gamma / N)
                    temp_weight = winner.weight * np.exp(k * self.gamma * temp_utility / (N * temp_probability))
                    winner.weight = temp_weight
            
            total_utility = 0 
            
            return temp_winner_list, total_utility
            

        
    def DependentRound(self, worker_list, k):
            temp_probability_list = copy.deepcopy([i.probability for i in worker_list])
            while any(0 < i < 1 for i in temp_probability_list):
                # temp_index_list 装概率对应在temp_probability_list中的索引
                temp_index_list = [index for index, element in enumerate(temp_probability_list) if element<1 and element>0]
                # temp_list 装概率
                temp_list=[i for i in temp_probability_list if i>0 and i<1]
                # print("*******DependentRound*******:len(temp_list):",len(temp_list))
                if len(temp_list) >= 2:
                    # 随机选择两个temp_list的索引，同时也是temp_index_list的索引
                    selected_indices = random.sample(range(len(temp_list)), 2)
                    # 选择两个索引对应的probability
                    workers = [temp_list[x] for x in selected_indices]
                    # 选择两个clients ,这里work代表概率
                    worker_1 = workers[0]
                    worker_2 = workers[1]
                    delta_1 = min([1 - worker_1, worker_2])
                    delta_2 = min([worker_1, 1 - worker_2])
                    # assign values based on probability
                    corn = np.random.uniform(0, 1)
                    if corn < delta_2 / (delta_1 + delta_2):
                        worker_1, worker_2 = worker_1 + delta_1, worker_2 - delta_1
                    else:
                        worker_1, worker_2 = worker_1 - delta_2, worker_2 + delta_2
                    index_1 = temp_index_list[selected_indices[0]]
                    index_2 = temp_index_list[selected_indices[1]]
                    temp_probability_list[index_1] = worker_1
                    temp_probability_list[index_2] = worker_2
                if len(temp_list) == 1:
                    if len([i for i in temp_probability_list if i == 1]) < k:
                        for i in range(len(temp_probability_list)):
                            if temp_probability_list[i]>0 and temp_probability_list[i]<1 :
                                temp_probability_list[i] = 1
                                break
                        break
                    else:
                        break
            selected_index = [index for index, element in enumerate(temp_probability_list) if element == 1]
            
            return [worker_list[i] for i in selected_index]
        
    def findAlpha(self,temp_list,delta):
            alpha=0
            error=0
            alpha_step=10
            weight_list=sorted(temp_list)
            for w in weight_list:
                if w==0:
                    continue
                if (len([i for i in weight_list if i > w]) * w + np.sum([i for i in weight_list if i <= w]))==0:
                    print(f'Error of findAlpha 1. len(weight_list):{len(weight_list)}, w:{w}')
                    # print(weight_list)
                    temp = w / (len([i for i in weight_list if i > w]) * w + np.sum([i for i in weight_list if i <= w]))
                else:
                    temp = w / (len([i for i in weight_list if i > w]) * w + np.sum([i for i in weight_list if i <= w]))
                if temp <= delta:
                    alpha = w
                if temp > delta:
                    values = [i for i in np.linspace(alpha, w, alpha_step)]
                    value_dic = {}
                    for value in values:
                        if  (len([i for i in weight_list if i > value]) * value + np.sum([i for i in weight_list if i <= value])) == 0:
                            print(f'Error of findAlpha 1. len(values):{len(values)}')
                            temp_value = value / (len([i for i in weight_list if i > value]) * value + np.sum(
                                [i for i in weight_list if i <= value]))
                        else:
                            temp_value = value / (len([i for i in weight_list if i > value]) * value + np.sum(
                                [i for i in weight_list if i <= value]))
                        if delta >= temp_value:
                            value_dic[value] = temp_value
                        else:
                            break
                    alpha = max(value_dic, key=value_dic.get)
                    error = value_dic[alpha]
                    break
            self.alpha = alpha
            return alpha,error