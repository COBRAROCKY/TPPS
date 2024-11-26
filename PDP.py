import numpy as np

# require python 3.8+

def PDP_select(clients, loss_dp_dict, rate, current_round, total_round):
    
    epsilon_dict = {client.id: client.DP.dp_epsilon for client in clients}
    score_dict = {}
    
    beta1 = 0.9
    beta2 = 10
    
    for client in clients:
        idx = client.id 
        
    # 大的排前面
    epsilon_sort_dict = sorted(epsilon_dict.items(), key=lambda item: item[1], reverse=True)
    loss_dp_sort_dict = sorted(loss_dp_dict.items(), key=lambda item: item[1], reverse=True)

    # 根据排序后的字典获得排名
    rank_epsilon = [id for id, epsilon in epsilon_sort_dict]
    rank_loss = [id for id, epsilon in loss_dp_sort_dict]

    for client in clients:
        idx = client.id
        score_dict[idx] = rank_epsilon.index(idx) + rank_loss.index(idx) * (beta1 ** ((total_round - current_round)/beta2))
    
    score_rank_dict = sorted(score_dict.items(), key=lambda item:item[1], reverse=False)
    winner = [rank for rank, score in score_rank_dict]   
    winner = winner[:int(rate*len(clients))]
    
    return winner
    
    
# for test


if __name__ == "__main__":
    class client:
        def __init__(self,id, dp_epsilon):
            self.id = id
            self.dp_epsilon = dp_epsilon
            
    clients = [client(1,2), client(3,4), client(5,6), client(7,8)]
    winner = PDP_select(clients, {1:0.1,3:0.9,5:0.2,7:0.5}, 0.5, 20, 40 )
    print(winner)