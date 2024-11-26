# from dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA
from dp.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple
import numpy as np
import torch



class DP(object):
    def __init__(self,
                 dp_mechanism,
                 dp_epsilon,
                 times,
                 dp_delta,
                 min_idx_sample=None,
                 dp_sample=None,
                 learning_rate=None,
                 dp_clip=None,
                 device=None,
                 using_dp=False,
        ):
        self.dp_mechanism = dp_mechanism
        self.dp_epsilon = dp_epsilon
        self.times = times 
        self.dp_delta = dp_delta
        self.min_idx_sample = min_idx_sample
        self.dp_sample = dp_sample
        
        self.lr = learning_rate 
        self.dp_clip = dp_clip 
        self.device = device
        # 这里预设dp_epsilon给的为全局隐私预算
        self.noise_scale = self.calculate_noise_scale()
        self.using_dp = using_dp
        
        # MAB用
        self.sensitivity = 0 
    
    def refresh_lr(self,learning_rate):
        self.lr = learning_rate 


    # self需要有dp_epsilon times 
    # Guassian: dp_epsilon times delta 
    # Laplace: dp_epsilon times
    # MA: dp_epsilon times delta dp_sample 
    # 建议self为client对象或者edge对象
    
    def calculate_noise_scale(self):
        if self.dp_mechanism == 'Laplace':
            # epsilon_single_query = self.dp_epsilon / self.times
            epsilon_single_query = self.dp_epsilon
            return Laplace(epsilon=epsilon_single_query)
        elif self.dp_mechanism == 'Gaussian':
            # epsilon_single_query = self.dp_epsilon / self.times
            epsilon_single_query = self.dp_epsilon 
            # delta_single_query = self.dp_delta / self.times
            delta_single_query = self.dp_delta 
            return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)
        elif self.dp_mechanism == 'MA':
            # TODO MA部分待处理
            print("待处理")
            # return Gaussian_MA(epsilon=self.args.dp_epsilon, delta=self.args.dp_delta, q=self.args.dp_sample, epoch=self.times)
            
    # self: lr dp_clip idxs_sample noise_scale device 

    def add_noise(self, model):
        # 计算灵敏度
        sensitivity = cal_sensitivity(self.lr, self.dp_clip, self.min_idx_sample)
        self.sensitivity = sensitivity
        # 获得模型的所有参数
        state_dict = model.state_dict()
        if self.dp_mechanism == 'Laplace':
            with torch.no_grad():
                for param in model.parameters():
                    param_device = param.device
                    noise = torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity / self.dp_epsilon, size=param.shape)).to(param_device)
                    param.data.add_(noise)    
                    #param.data.add_(torch.randn_like(param.data) * sensitivity / self.noise_scale)
                
        elif self.dp_mechanism == 'Gaussian':
            # for k, v in state_dict.items():
            #     state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
            #                                                         size=v.shape)).to(self.device)
            with torch.no_grad():
                for param in model.parameters():
                    param.data.add_(torch.randn_like(param.data) * sensitivity * self.noise_scale)
            
            
        elif self.dp_mechanism == 'MA':
            sensitivity = cal_sensitivity_MA(self.lr, self.dp_clip, self.idxs_sample)
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                    size=v.shape)).to(self.device)
        model.load_state_dict(state_dict)
        # print("adding noise finished")
            
            
            
    # 梯度裁剪选择
    # self: dp_mechanism  dp_clip
    def clip_gradients(self,model):
        if self.dp_mechanism == 'Laplace':
            self.per_sample_clip(model,self.dp_clip,norm=2)
            # print("待实现")
        elif self.dp_mechanism == 'Gaussian' or self.dp_mechanism == 'MA':
            self.per_sample_clip(model,self.dp_clip,norm=2)

    # 梯度裁剪
    # self: 
    def per_sample_clip(self, model, clipping, norm):
        
        # * 旧 无法适应CIFAR-10
        # grad_samples = [x.grad for x in model.parameters()]
        # per_param_norms = [
        #     g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        # ]
        # per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        # per_sample_clip_factor = (
        #     torch.div(clipping, (per_sample_norms + 1e-6))
        # ).clamp(max=1.0)
        # for grad in grad_samples:
        #     factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
        #     grad.detach().mul_(factor.to(grad.device))
        
        # *新
        total_norm = 0.0
        for param in model.parameters():
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        with torch.no_grad():
            for param in model.parameters():
                # 计算裁剪阈值
                threshold = max(total_norm / clipping , 1)
                # 裁剪参数
                param.data.clamp_(-threshold, threshold)
        
        
        
        # 对每一层参数的梯度进行平均
        # with torch.no_grad():
        #     for name, module in model.named_children():
        #         # 获取当前层的所有参数
        #         params = list(module.parameters())
        #         # 计算当前层参数的梯度的平均值
        #         average_grad = torch.mean(torch.stack([param.grad for param in params]), dim=0)
        #         # 将平均梯度赋给当前层的所有参数
        #         for param in params:
        #             param.grad = average_grad.clone()
        
        # 打印用
        # for grad in grad_samples:
        #     print("the grad:",grad)
        #     print("the Shape:",grad.shape)
        
        # average per sample gradient after clipping and set back gradient
        # for param in model.parameters():
        #     # 这里按原有的方法，出现different size问题，少了第一维,即批次大小 (logistic模型下)
        #     # param.grad = param.grad.detach().mean(dim=0)
        #     print("name:",param.name," ,data:",param.data)
            
        #     # 计算梯度的均值，并对梯度进行克隆
        #     temp_grad = param.grad.detach().mean(dim=0).clone()
            
        #     # 创建一个新的梯度张量，用于存储修改后的梯度
        #     new_grad = torch.zeros_like(param.grad)
            
        #      # 检查梯度的形状是否与原始梯度的形状相同
        #     if temp_grad.shape != param.grad.shape:
        #         # 如果均值梯度是标量，对原始梯度进行标量除法
        #         if temp_grad.ndim == 0:
        #             new_grad = param.grad / temp_grad
        #         # 如果均值梯度是二维的，则将其扩展到与原始梯度相同的形状
        #         else:
        #             print("tempSharp:", temp_grad.shape, " paramSharp", param.grad.shape)
        #             new_grad = temp_grad.expand_as(param.grad)
        #     else:
        #         # 如果形状相同，则直接赋值
        #         new_grad = temp_grad
            
        #     # 将新的梯度赋给 param.grad
        #     param.grad = new_grad
        
        
    def add_noise_Lap(self, parameter, sensitivity, budget):
        # 为parameter加上拉普拉斯噪声
        noise = np.random.laplace(loc=0, scale=sensitivity/budget)
        parameter = parameter + noise 
        return parameter, noise 
        
