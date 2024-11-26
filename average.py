import copy
import torch
from torch import nn

def average_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    # temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    
        
    with torch.no_grad():
        for i,w_i in enumerate(w):
            if i==0 :
                break
            for param in w_i.parameters():
                if param.requires_grad:
                    w_avg.state_dict()[param.name] += (
                        s_num[i] / total_sample_num) * param.data
    
    
    return w_avg