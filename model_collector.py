import os
import torch
import random

class ModelCollector:
    def __init__(self, args, clients):
        self.args = args
        self.clients = clients
        self.save_rounds = [1, args.num_communication // 2, args.num_communication]
        self.save_dir = 'tracked_models'
        os.makedirs(self.save_dir, exist_ok=True)
        
    def initialize_dp_tracking(self):
        """初始化DP模式下的追踪（epsilon=2,5,10的客户端）"""
        tracked_clients = []
        target_epsilons = [2, 5, 10]
        
        for target_epsilon in target_epsilons:
            for client in self.clients:
                if client.DP.dp_epsilon == target_epsilon:
                    tracked_clients.append(client.id)
                    print(f"找到目标客户端: id={client.id}, epsilon={target_epsilon}")
                    break
        
        if len(tracked_clients) != 3:
            print("Warning：未找到所有目标epsilon值的客户端！")
            print(f"当前找到的客户端: {tracked_clients}")
            
        return tracked_clients
    
    def initialize_random_tracking(self):
        """初始化随���追踪模式"""
        random_client_id = random.randint(0, len(self.clients) - 1)
        print(f"随机选择追踪的客户端: id={random_client_id}")
        return random_client_id
    
    def save_dp_models(self, tracked_clients, edges, current_round):
        """保存DP模式下的模型"""
        if current_round in self.save_rounds:
            stage = 1 if current_round == 1 else (2 if current_round == self.args.num_communication // 2 else 3)
            
            for client_id in tracked_clients:
                for edge in edges:
                    if client_id in edge.cids:
                        client = next((c for c in self.clients if c.id == client_id), None)
                        if client is not None:
                            save_path = os.path.join(
                                self.save_dir, 
                                f'dataset={self.args.dataset}_id={client_id}_epsilon={client.DP.dp_epsilon}_第{stage}阶段.pth'
                            )
                            torch.save(client.model.shared_layers.state_dict(), save_path)
                            print(f'已保存客户端{client_id}(epsilon={client.DP.dp_epsilon})在第{stage}阶段的模型到{save_path}')
    
    def save_random_model(self, random_client_id, edges, current_round):
        """随机追踪模式下的模型"""
        if current_round in self.save_rounds:
            stage = 1 if current_round == 1 else (2 if current_round == self.args.num_communication // 2 else 3)
            
            for edge in edges:
                if random_client_id in edge.cids:
                    client = next((c for c in self.clients if c.id == random_client_id), None)
                    if client is not None:
                        save_path = os.path.join(
                            self.save_dir, 
                            f'dataset={self.args.dataset}_random_id={random_client_id}_第{stage}阶段.pth'
                        )
                        torch.save(client.model.shared_layers.state_dict(), save_path)
                        print(f'已保存随机选择的客户端{random_client_id}在第{stage}阶段的模型到{save_path}') 