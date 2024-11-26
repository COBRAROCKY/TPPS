import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    #dataset and model
    parser.add_argument(
        '--remark',
        default="exp-0", 
        type=str,
        help= "experiment remark like 'exp-1' or 'exp-2' "
    )
    parser.add_argument(
        '--dataset',
        type = str,
        default = 'cifar10',
        help = 'name of the dataset: mnist, cifar10'
    )
    parser.add_argument(
        '--model',
        type = str,
        default = 'cnn',
        help='name of model. mnist: logistic, lenet; cifar10: cnn_tutorial, cnn_complex'
    )
    parser.add_argument(
        '--input_channels',
        type = int,
        default = 3,
        help = 'input channels. mnist:1, cifar10 :3'
    )
    parser.add_argument(
        '--output_channels',
        type = int,
        default = 10,
        help = 'output channels'
    )
    #nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 10,
        help = 'batch size when trained on client'
    )
    parser.add_argument(
        '--num_communication',
        type = int,
        default=1,
        help = 'number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=1,
        help='number of local update (tau_1)'
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type = int,
        default=1,
        help = 'number of edge aggregation (tau_2)'
    )
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.001,
        help = 'learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--lr_decay',
        type = float,
        default= '1',
        help = 'lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type = int,
        default=1,
        help= 'lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type = float,
        default = 0,
        help = 'SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type = float,
        default = 0,
        help= 'The weight decay rate'
    )
    parser.add_argument(
        '--verbose',
        type = int,
        default = 0,
        help = 'verbose for print progress bar'
    )
    #setting for federeated learning
    parser.add_argument(
        '--iid',
        type = int,
        default = 0,
        help = 'distribution of the data, 1,0 (SuperIID mode, client labels, data, etc. are very uneven), -2 (one-class), -3 (Delicacy distribution mode, need to specify alpha_class and alpha_data)'
    )
    parser.add_argument(
        '--edgeiid',
        type=int,
        default=0,
        help='distribution of the data under edges, 1 (edgeiid),0 (edgeniid) (used only when iid = -2)'
    )
    parser.add_argument(
        '--frac',
        type = float,
        default = 1,
        help = 'fraction of participated clients'
    )
    parser.add_argument(
        '--num_clients',
        type = int,
        default = 10,
        help = 'number of all available clients'
    )
    parser.add_argument(
        '--num_edges',
        type = int,
        default= 1,
        help= 'number of edges'
    )
    parser.add_argument(
        '--seed',
        type = int,
        default = 1,
        help = 'random seed (defaul: 1)'
    )
    parser.add_argument(
        '--dataset_root',
        type = str,
        default = 'data',
        help = 'dataset root folder'
    )
    parser.add_argument(
        '--show_dis',
        type= int,
        default= 0,
        help='whether to show distribution'
    )
    parser.add_argument(
        '--classes_per_client',
        type=int,
        default = 2,
        help='under artificial non-iid distribution, the classes per client'
    )
    parser.add_argument(
        '--gpu',
        type = int,
        default=0,
        help = 'GPU to be selected, 0, 1, 2, 3'
    )

    parser.add_argument(
        '--mtl_model',
        default=0,
        type = int
    )
    parser.add_argument(
        '--global_model',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--local_model',
        default=0,
        type=int,
        help="Number of client local updates"
    )
    parser.add_argument(
        '--alg',
        default="TPPS",
        type=str,
        help="Client selection algorithm"
    )
    
    # DP参数
    parser.add_argument(
        '--using_dp',
        default=0, 
        type=int,
        help="Controls whether or not noise is added to models"
    )
    parser.add_argument(
        '--dp_mechanism',
        default=None, # Guassian / Laplace / MA
        type=str,
        help="Guassian / Laplace / MA"
    )
    parser.add_argument(
        '--dp_epsilon',
        default=None, 
        type=float,
        help="Privacy Budget"
    )
    parser.add_argument(
        '--dp_delta',
        default=None, 
        type=float,
        help="Probability of tolerating the destruction of the original ε-dp in Gaussian noise addition, preferably 1/dataset_size Probability of tolerating the destruction of the original ε-dp in Gaussian noise addition, preferably 1/dataset_size"
    )
    parser.add_argument(
        '--dp_clip',
        default=None, 
        type=float,
        help="gradient clipping threshold"
    )
    parser.add_argument(
        '--loss_dp_eval',
        default=0,
        type=int,
        help="Whether to evaluate the model after noise addition, the loss loss measured on the client's local dataset, i.e., loss_dp, is required for the PDP algorithm using the"
    )
    parser.add_argument(
        '--mab_rate',
        default=None,
        type=float,
        help="MAB Selection Algorithm for Harvesting Client Sample Rates"
    )
    parser.add_argument(
        '--random_rate',
        default=None,
        type=float,
        help="Client Sampling Rate in Randomized Selection"
    )
    # PDP benchmark 设置
    parser.add_argument(
        '--is_select_after_train',
        default=0,
        type=int,
        help="Whether the algorithm used is trained and then client-selected,1,0"
    )
    parser.add_argument(
        '--pdp_rate',
        default=None,
        type=float,
        help="Sampling rate of the PDP algorithm"
    )
    parser.add_argument(
        '--alpha_data',
        default=None,
        type=float,
        help="Dirichlet distribution parameter, the distribution of data, the larger the more uniform, the smaller the data volume the greater the variability"
    )
    parser.add_argument(
        '--alpha_class',
        default=None,
        type=float,
        help="Delicacy distribution parameter, the distribution of labels, larger is more homogeneous, smaller is more heterogeneous"
    )

    parser.add_argument(
        '--is_noise_labels',
        default=0,
        type=int,
        help='Whether to emulate noise_labele, default label error rate per client sampled in a uniform distribution of 0-1'
    )

    parser.add_argument(
        '--TPPS_rate',
        default=0,
        type=float,
        help='The rate of sampled clients in the TPPS algorithm, k = TPPS_rate * N'
    )
    
    parser.add_argument(
        '--Poisson_lambda',
        default=0,
        type=float,
        help='Poisson selection strategy: Client sample probability'
    )

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
