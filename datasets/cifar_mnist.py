"""
download the required dataset, split the data among the clients, and generate DataLoader for training
"""
import os
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import random 

import torch
import torch.backends.cudnn as cudnn
cudnn.banchmark = True

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from options import args_parser

import ssl
ssl._create_default_https_context = ssl._create_unverified_context



class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, target = self.dataset[self.idxs[item]]
        return image, target

def gen_ran_sum(_sum, num_users):
    base = 100*np.ones(num_users, dtype=np.int32)
    _sum = _sum - 100*num_users
    p = np.random.dirichlet(np.ones(num_users), size=1)
    print(p.sum())
    p = p[0]
    size_users = np.random.multinomial(_sum, p, size=1)[0]
    size_users = size_users + base
    print(size_users.sum())
    return size_users

def get_mean_and_std(dataset):
    """
    compute the mean and std value of dataset
    """
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("=>compute mean and std")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def iid_esize_split(dataset, args, kwargs, is_shuffle = True):
    """
    split the dataset to users
    Return:
        dict of the data_loaders
    """
    sum_samples = len(dataset)
    num_samples_per_client = int(sum_samples / args.num_clients)
    # change from dict to list
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(args.num_clients):
        dict_users[i] = np.random.choice(all_idxs, num_samples_per_client, replace = False)
        #dict_users[i] = dict_users[i].astype(int)
        #dict_users[i] = set(dict_users[i])
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = args.batch_size,
                                    shuffle = is_shuffle, **kwargs)

    return data_loaders

def iid_nesize_split(dataset, args, kwargs, is_shuffle = True):
    sum_samples = len(dataset)
    num_samples_per_client = gen_ran_sum(sum_samples, args.num_clients)
    # change from dict to list
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for (i, num_samples_client) in enumerate(num_samples_per_client):
        dict_users[i] = np.random.choice(all_idxs, num_samples_client, replace = False)
        #dict_users[i] = dict_users[i].astype(int)
        #dict_users[i] = set(dict_users[i])
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = args.batch_size,
                                    shuffle = is_shuffle, **kwargs)

    return data_loaders

def niid_esize_split(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0] * args.num_clients
    # each client has only two classes of the network
    num_shards = 2* args.num_clients
    # the number of images in one shard 每一个切片，图片的数量
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    # is_shuffle is used to differentiate between train and test
    if is_shuffle:
        
        labels = dataset.train_labels
    
    else:
    
        labels = dataset.test_labels
    
    # 标签和数据索引对齐
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    # sort the data according to their label
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    #divide and assign
    for i in range(args.num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace= False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = args.batch_size,
                                    shuffle = is_shuffle, **kwargs)
    return data_loaders

def niid_esize_split_train(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0]* args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
#     no need to judge train ans test here
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
#     divide and assign
#     and record the split patter
    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace= False)
        split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, split_pattern

def niid_esize_split_test(dataset, args, kwargs, split_pattern,  is_shuffle = False ):
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    #     no need to judge train ans test here
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
#     divide and assign
    for i in range(args.num_clients):
        rand_set = split_pattern[i][0]
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, None

def niid_esize_split_train_large(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0]* args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace= False)
        # split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
            # store the label
            split_pattern[i].append(dataset.__getitem__(idxs[rand * num_imgs])[1])
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, split_pattern

def niid_esize_split_test_large(dataset, args, kwargs, split_pattern, is_shuffle = False ):
    """
    :param dataset: test dataset
    :param args:
    :param kwargs:
    :param split_pattern: split pattern from trainloaders
    :param test_size: length of testloader of each client
    :param is_shuffle: False for testloader
    :return:
    """
    data_loaders = [0] * args.num_clients
    # for mnist and cifar 10, only 10 classes
    num_shards = 10
    num_imgs = int (len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(len(dataset))
    #     no need to judge train ans test here
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
#     divide and assign
    for i in range(args.num_clients):
        rand_set = split_pattern[i]
        # idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, None

def niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0] * args.num_clients
    #one class perclients
    #any requirements on the number of clients?
    # 切片数量
    num_shards = args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    # 切片的索引
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    # 图片索引
    idxs = np.arange(num_shards * num_imgs)
    if is_shuffle:
        # 所有标签， 按MNIST样本原来数据排序
        labels = dataset.train_labels
    else:
        labels = dataset.test_labels
    # 讲标签从小到大排好，并且将图片索引对齐
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    #divide and assign
    for i in range(args.num_clients):
        rand_set = set(np.random.choice(idx_shard, 1, replace = False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand+1)*num_imgs]), axis = 0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                            batch_size = args.batch_size,
                            shuffle = is_shuffle, **kwargs)
    return data_loaders

# * 以下代码为 Super non-iid （By MJ) 每个客户端分配至少一个类，余下随机分(每次多个类)
def super_niid_esize_split(dataset, args, kwargs, is_shuffle=True):
    data_loaders = [0] * args.num_clients
    # Each client initially gets one shard 
    # x_shard 为切片对client的倍数
    if args.dataset == 'mnist':
        x_shard = 6 # mnist 
    elif args.dataset == 'cifar10':
        x_shard = 40 # cifar
    num_initial_shards = args.num_clients * x_shard
    # Number of shards initially assigned to each client
    num_initial_shards_per_client = 1
    # The number of images in one shard
    num_imgs = int(len(dataset) / num_initial_shards)
    idx_shard = [i for i in range(num_initial_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    # 每个图片的索引
    idxs = np.arange(num_initial_shards * num_imgs)
    if is_shuffle:
        # 所有标签， 按MNIST样本原来数据排序, 只需要样本数idx一样的标签数目 使用不到全部图也无所谓
        try:
            labels = dataset.train_labels[:len(idxs)]
        except AttributeError:
            labels = dataset.targets[:len(idxs)]
    else:
        try:
            labels = dataset.test_labels[:len(idxs)]
        except AttributeError:
            labels = dataset.targets[:len(idxs)] 
            
    # 讲标签从小到大排好，并且将图片索引对齐
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    
    # Assign one shard to each client initially
    for i in range(args.num_clients):
        rand = np.random.choice(idx_shard, 1, replace=False)[0]
        idx_shard.remove(rand)
        dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
        dict_users[i] = dict_users[i].astype(int)

    # Assign remaining shards randomly to clients until all shards are allocated
    while idx_shard:
        # ! 旧：8，10
        # ! 新：2，3
        num_shards_to_assign = np.random.randint(1, 2)  # Randomly assign 1 to 3 shards
        # num_clients_with_space = sum(len(dict_users[i]) < num_initial_shards_per_client * num_imgs for i in range(args.num_clients))
        # if num_clients_with_space == 0:
        #     break  # No clients have space left to assign shards
        shards_to_assign = np.random.choice(idx_shard, min(num_shards_to_assign,len(idx_shard)), replace=False)
        # Randomly choose a client
        rand_client = random.randint(0, args.num_clients-1)
        for shard in shards_to_assign:
            # Randomly choose a client with space and assign the shard
            # client_with_space = np.random.choice([i for i in range(args.num_clients) if len(dict_users[i]) < num_initial_shards_per_client * num_imgs], 1)[0]
            dict_users[rand_client] = np.concatenate((dict_users[rand_client], idxs[shard * num_imgs: (shard + 1) * num_imgs]), axis=0)
            dict_users[rand_client] = dict_users[rand_client].astype(int)
            idx_shard.remove(shard)

    # # Create DataLoader for each client
    for i in range(args.num_clients):
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle, **kwargs)

    return data_loaders


def dirichlet_split(dataset,args, kwargs, is_shuffle=True):
    data_loaders = [0] * args.num_clients
    alpha_data = args.alpha_data
    alpha_class =  args.alpha_class
    n_clients = args.num_clients
    
    # 3. 确定类别数
    n_classes = len(dataset.classes)

    # 4. 生成数据量分布
    total_samples = len(dataset)  # 假设总样本数为数据集中的样本数
    data_amounts = np.random.dirichlet(np.array([alpha_data] * n_clients), size=1)[0]

    # 5. 划分数据集并创建客户端数据集
    client_datasets = []
    client_labels_list = []  # 存储每个客户端的标签列表

    # 辅助函数，用于获取数据集中的所有标签
    def get_all_labels(dataset):
        return [label for _, label in enumerate(dataset.targets)]

    all_labels = get_all_labels(dataset)
    unique_labels, inverse = np.unique(all_labels, return_inverse=True)

    # 为每个类别生成索引列表
    class_indices = [np.where(inverse == label)[0] for label in unique_labels]

    def sample_from_class(class_indices, client_total_samples, alpha):
        dirichlet_params = np.random.dirichlet(np.array([alpha] * len(class_indices)))
        num_samples_per_class = [min(int(np.floor(param * client_total_samples)), len(indices)) for param, indices in zip(dirichlet_params, class_indices)]
        client_indices_class = []
        for num_samples, indices in zip(num_samples_per_class, class_indices):
            client_indices_class.extend(np.random.choice(indices, num_samples, replace=False))
        return np.array(client_indices_class)

    for i in range(n_clients):
        client_total_samples = int(np.floor(total_samples * data_amounts[i]))  # 根据狄利克雷分布确定每个客户端的样本总数

        # 为每个类别生成一个索引数组
        class_indices = [np.where(inverse == label)[0] for label in unique_labels]

        # 为每个客户端从每个类别中采样
        client_indices = sample_from_class(class_indices, client_total_samples, alpha_class)


        data_loaders[i] = DataLoader(DatasetSplit(dataset, client_indices),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle, **kwargs)
        
        # 创建Subset对象
        # client_dataset = torch.utils.data.Subset(dataset, client_indices)
        # client_datasets.append(client_dataset)
        
        # 收集客户端的标签
        # client_labels = [all_labels[idx] for idx in client_indices]
        # client_labels_list.append(client_labels)

    # print(client_labels_list)
    
    
    # 6. 打印每个客户端的类别分布情况和样本总数
    for i, client_labels in enumerate(client_labels_list):
        print(f"Client {i} class distribution: {np.round(np.bincount(client_labels)/len(client_labels),3)}   num:{len(client_labels)}")
        
        
    # for i, client_labels in enumerate(client_labels_list):
    #     # print(f"Client {i} client_labels {client_labels}")
    #     data_loaders[i] = DataLoader(DatasetSplit(dataset, client_labels),
    #                                  batch_size=args.batch_size,
    #                                  shuffle=is_shuffle, **kwargs)
    return data_loaders

def split_data(dataset, args, kwargs, is_shuffle = True):
    """
    return dataloaders
    """
    if args.iid == 1:
        data_loaders = iid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == 0:
        # 原有的 non-iid
        # data_loaders = niid_esize_split(dataset, args, kwargs, is_shuffle)
        # 修改的 super non-iid (按照切片随机分)
        data_loaders = super_niid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == -1:
        data_loaders = iid_nesize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == -2:
        # 将数据集划分为非重叠的子集（dataloader),其中每个客户端仅接收来自一个类的数据
        data_loaders = niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle)
    elif args.iid == -3:
        # 将数据集的数量、类别各安排一个狄利克雷参数控制其分布异构程度
        data_loaders = dirichlet_split(dataset, args, kwargs, is_shuffle)
    else :
        raise ValueError('Data Distribution pattern `{}` not implemented '.format(args.iid))
    return data_loaders

def get_dataset(dataset_root, dataset, args):
    trains, train_loaders, tests, test_loaders = {}, {}, {}, {}
    if dataset == 'mnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_mnist(dataset_root, args)
    elif dataset == 'cifar10':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar10(dataset_root, args)
    elif dataset == 'femnist':
        raise ValueError('CODING ERROR: FEMNIST dataset should not use this file')
    elif dataset == 'cifar100':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar100(dataset_root, args)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders, v_train_loader, v_test_loader


"""
get_mnist方法用于获取MNIST数据集，并将其划分为训练集和测试集的dataloader。

参数:
- dataset_root: 数据集的根目录
- args: 参数对象，包含各种配置选项

返回:
- train_loaders: 训练集的dataloader列表
- test_loaders: 测试集的dataloader列表
- v_train_loader: 虚拟训练集的dataloader
- v_test_loader: 虚拟测试集的dataloader

步骤:
1. 检查是否使用CUDA加速，并设置相应的kwargs参数。
2. 定义数据预处理的transform，包括将图像转换为张量和标准化。
3. 下载并加载MNIST训练集和测试集。
4. 调用split_data方法，将训练集和测试集划分为多个客户端的数据加载器。
5. 创建虚拟训练集和测试集的数据加载器，用于计算整个数据集的梯度和准确性。
6. 返回训练集、测试集和虚拟数据集的dataloader。
"""


def get_mnist(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
    transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ])
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train = True,
                            download = True, transform = transform)
    test =  datasets.MNIST(os.path.join(dataset_root, 'mnist'), train = False,
                            download = True, transform = transform)
    #note: is_shuffle here also is a flag for differentiating train and test
    train_loaders = split_data(train, args, kwargs, is_shuffle = True)
    test_loaders = split_data(test,  args, kwargs, is_shuffle = False)
    #the actual batch_size may nee3d to change.... Depend on the actual gradient...
    #originally written to get the gradient of the whole dataset
    #but now it seems to be able to improve speed of getting accuracy of virtual sequence
    v_train_loader = DataLoader(train, batch_size = args.batch_size * args.num_clients,
                                shuffle = True, **kwargs)
    v_test_loader = DataLoader(test, batch_size = args.batch_size * args.num_clients,
                                shuffle = False, **kwargs)
    return  train_loaders, test_loaders, v_train_loader, v_test_loader


def get_cifar10(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory':True} if is_cuda else{}
    if args.model == 'cnn_complex':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.model == 'resnet18':
        transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding = 4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError("this nn for cifar10 not implemented")
    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train = True,
                        download = True, transform = transform_train)
    test = datasets.CIFAR10(os.path.join(dataset_root,'cifar10'), train = False,
                        download = True, transform = transform_test)
    v_train_loader = DataLoader(train, batch_size = args.batch_size,
                                shuffle = True, **kwargs)
    v_test_loader = DataLoader(test, batch_size = args.batch_size,
                                shuffle = False, **kwargs)
    train_loaders = split_data(train, args, kwargs, is_shuffle = True)
    test_loaders = split_data(test,  args, kwargs, is_shuffle = False)
    return  train_loaders, test_loaders, v_train_loader, v_test_loader


def get_cifar100(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
    
    # 定义CIFAR-100的transform
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    
    # 下载并加载CIFAR-100数据集
    train = datasets.CIFAR100(os.path.join(dataset_root, 'cifar100'), train=True,
                              download=True, transform=transform_train)
    test = datasets.CIFAR100(os.path.join(dataset_root, 'cifar100'), train=False,
                             download=True, transform=transform_test)
    
    # 创建DataLoader
    v_train_loader = DataLoader(train, batch_size=args.batch_size,
                                shuffle=True, **kwargs)
    v_test_loader = DataLoader(test, batch_size=args.batch_size,
                               shuffle=False, **kwargs)
    
    # 使用split_data函数将数据集划分为多个客户端的数据加载器
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)
    test_loaders = split_data(test, args, kwargs, is_shuffle=False)
    
    return train_loaders, test_loaders, v_train_loader, v_test_loader

def show_distribution(dataloader, args):
    """
    show the distribution of the data on certain client with dataloader
    return:
        percentage of each class of the label
    """
    if args.dataset == 'mnist':
        try:
            # dataloader.dataset.dataset ==> MNIST 直接拿到全部标签
            labels = dataloader.dataset.dataset.train_labels.numpy()
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.test_labels.numpy()
        # labels = dataloader.dataset.dataset.train_labels.numpy()
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        try:
            labels = dataloader.dataset.dataset.targets
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.targets
        # labels = dataloader.dataset.dataset.train_labels
    elif args.dataset == 'fsdd':
        labels = dataloader.dataset.labels
    else:
        raise ValueError("`{}` dataset not included".format(args.dataset))
    num_samples = len(dataloader.dataset)
    # print(num_samples)
    idxs = [i for i in range(num_samples)]
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    distribution = [0] * len(unique_labels)
    for idx in idxs:
        img, label = dataloader.dataset[idx]
        distribution[label] += 1
    distribution = np.array(distribution)
    distribution = distribution / num_samples
    return distribution

if __name__ == '__main__':
    args = args_parser()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loaders, test_loaders, _, _ = get_dataset(args.dataset_root, args.dataset, args)
    print(f"The dataset is {args.dataset} divided into {args.num_clients} clients/tasks in an iid = {args.iid} way")
    for i in range(args.num_clients):
        train_loader = train_loaders[i]
        print(len(train_loader.dataset))
        distribution = show_distribution(train_loader, args)
        print("dataloader {} distribution".format(i))
        print(distribution)

