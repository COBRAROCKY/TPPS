
# **TPPS Algorithm under Hierarchical Federated Learning**

***

*   The code for "Towards Data Privacy in Hierarchical Federated Learning: A Probabilistic Client Sampling Approach"

*   To build the program running environment, please use "environment.yml".

*   To run TPPS, you must make sure you have '--TPPS\_rate' , '--alg TPPS' and '--using\_dp' in your command line.&#x20;

*   If you want to reproduce a training process, you can use '--seed'.

*   For  details about the command line to run TPPS, please see 'options'.

# Example 1 : For running code with MNIST and LeNet (with DP) :

    python HFL.py  
    --dataset mnist 
    --model lenet 
    --num_clients  60 
    --num_edges 3 
    --frac 1
    --num_local_update 2 
    --num_edge_aggregation 5 
    --num_communication 80 
    --batch_size 16 
    --iid -3 
    --edgeiid 1 
    --show_dis 1 
    --lr 0.001 
    --lr_decay 0.995 
    --lr_decay_epoch 1 
    --momentum 0 
    --weight_decay 0 
    --using_dp 1 
    --dp_mechanism Gaussian 
    --dp_delta 0.001  
    --dp_clip 10   
    --alg TPPS 
    --TPPS_rate 0.4
    --alpha_data 10
    --alpha_class 10 
    --remark test
    --gpu 0

# Example 2 : For running code with CIFAR10 and Resnet18(with DP):

    python HFL.py  
    --dataset cifar10 
    --model resnet18     
    --num_clients 60 
    --num_edges 3 
    --frac 1
    --num_local_update 2 
    --num_edge_aggregation 5 
    --num_communication 160 
    --batch_size 8 
    --iid -3 
    --edgeiid 1 
    --show_dis 1 
    --lr 0.001 
    --lr_decay 0.995 
    --lr_decay_epoch 1 
    --momentum 0 
    --weight_decay 0 
    --using_dp 1 
    --dp_mechanism Laplace 
    --dp_delta 0.001 
    --dp_clip 10   
    --alg TPPS 
    --TPPS_rate 0.4
    --alpha_data 40
    --alpha_class 10 
    --remark test
    --gpu 0

Author：Junjie Mai & hengzhi Wang
