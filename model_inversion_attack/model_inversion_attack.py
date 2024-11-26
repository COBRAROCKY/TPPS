import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from train_lenet import LeNet as Eval_LeNet
import os
import torch.nn.functional as F
from torchvision.datasets import EMNIST
from train_gan import Generator, Critic

import random 
import numpy as np
import time 


    # 以下为HFL中使用的LeNet
class LeNet(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size= 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50, output_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return  x


class ModelInversionAttack:
    def __init__(self, model_path, target_label, attacker=None, discriminator=None, evaluator=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = LeNet(1,10).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.model.eval()
        self.target_label = target_label
        
        self.attacker = attacker  # Generator模型
        self.attacker.eval()  # 确保生成器在评估模式
        self.discriminator = discriminator
        self.discriminator.eval() 
        self.evaluator = evaluator 
        self.evaluator.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.inverse_transform = transforms.Compose([
            transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))
        ])
        
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]

    def total_variation_loss(self, x):
        """计算总变差损失"""
        # 计算水平和垂直方向的差异
        diff_i = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))  # 水平方向
        diff_j = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))  # 垂直方向
        return diff_i + diff_j

    def attack(self, num_iterations=1000, save_interval=10):
        """执行增强的模型反转攻击"""
        os.makedirs('inversion_results', exist_ok=True)
        
        # 初始化多个潜在向量以避免batch size=1的问题
        batch_size = 64  # 使用大于1的batch size
        latent_vectors = torch.randn(batch_size, 32, device=self.device)
        latent_vectors = nn.Parameter(latent_vectors, requires_grad=True)
        
        # 使用Adam优化器
        optimizer = optim.Adam([latent_vectors], lr=0.01)  # 降低学习率
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=0.001)
        
        target = torch.tensor([self.target_label] * batch_size, device=self.device)
        
        best_loss = float('inf')
        best_image = None
        
        print(f'开始对模型 {self.model_name} 的标签 {self.target_label} 进行增强反转攻击...')
        
        with torch.no_grad():
            # 初始测试生成器
            test_latent = torch.randn(batch_size, 32, device=self.device)
            try:
                test_output = self.attacker(test_latent)
                print(f"生成器测试成功，输出形状: {test_output.shape}")
            except Exception as e:
                print(f"生成器测试失败: {str(e)}")
                return None

        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # try:
            # 使用生成器生成图像
            with torch.set_grad_enabled(True):
                generated_images = self.attacker(latent_vectors)
            
            # 检查生成图像的形状
            if generated_images.shape[1:] != (1, 28, 28):
                raise ValueError(f"Generator output shape {generated_images.shape} is incorrect. Expected (N, 1, 28, 28)")
            
            # 应用变换
            x_processed = torch.stack([self.transform(torch.sigmoid(img)) for img in generated_images])
            output = self.model(x_processed)
            
            # 计算损失
            classification_loss = self.confidence_loss(output, target)
            reconstruction_loss = F.mse_loss(generated_images, x_processed)
            prior_loss = self.prior_loss(generated_images)
            feature_loss = self.feature_loss(generated_images)
            tv_loss = self.total_variation_loss(generated_images)
            l2_loss = torch.norm(generated_images)
            
            # 动态调整损失权重
            alpha = min(1.0, i / (num_iterations * 0.2))
            loss = (classification_loss + 
                    alpha * (0.1 * prior_loss + 
                            0.05 * feature_loss + 
                            0.001 * tv_loss + 
                            0.001 * l2_loss + 
                            0.1 * reconstruction_loss))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([latent_vectors], 1.0)
            optimizer.step()
            scheduler.step()

            
            # 更新最佳结果 - 只保存第一个图像
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_image = generated_images.clone()  # 
            
            # 定期保存结果
            if (i + 1) % save_interval == 0:
                print(f'Iteration {i+1}/{num_iterations}, '
                        f'Loss: {loss.item():.4f}, '
                        f'Classification Loss: {classification_loss.item():.4f}, '
                        f'Reconstruction Loss: {reconstruction_loss.item():.4f}')
                self.save_batch_image(torch.sigmoid(generated_images), i)  # 只保存第一个图像
                
            # except Exception as e:
            #     print(f"迭代 {i+1} 发生错误: {str(e)}")
            #     continue
        
        # 保存最终最佳结果
        if best_image is not None:
            print(f"Best image shape: {best_image.shape}")
            self.save_batch_image(torch.sigmoid(best_image),'BEST')
            print(f'完成对模型 {self.model_name} 标签 {self.target_label} 的攻击，最终损失: {best_loss:.4f}')
        else:
            print("未能生成有效的图像")
        
        return best_image

    def confidence_loss(self, output, target):
        """计算置信度损失"""
        target_probs = F.softmax(output, dim=1)
        target_prob = torch.mean(torch.gather(target_probs, 1, target.unsqueeze(1)))
        other_probs = torch.cat([output[:, :target[0]], output[:, target[0]+1:]], dim=1)
        margin_loss = torch.mean(torch.max(other_probs, dim=1)[0] - torch.gather(output, 1, target.unsqueeze(1)).squeeze())
        return -torch.log(target_prob) + 0.1 * margin_loss

    def feature_loss(self, x):
        """提取和匹配特征层的输出"""
        features = []
        x_processed = torch.stack([self.transform(torch.sigmoid(img)) for img in x])
        
        # 获取中间层特征
        x = self.model.conv1(x_processed)
        features.append(F.relu(x))
        # x = self.model.pool(features[-1])
        # features.append(x)
        x = self.model.conv2(x)
        features.append(F.relu(x))
        
        # 计算特征损失（促使特征图具有合理的统计特性）
        feature_loss = 0
        for feat in features:
            # 特征应该是稀疏的
            feature_loss += 0.1 * torch.mean(torch.abs(feat))
            # 特征图应该有合适的激活范围
            feature_loss += 0.1 * torch.mean(F.relu(feat - 2) + F.relu(-feat))
            
        return feature_loss

    def prior_loss(self, x):
        """添加先验知识约束"""
        # 稀疏性损失：鼓励图像中的大部分像素接近零
        sparsity_loss = torch.mean(torch.abs(x))
        
        # 平滑性损失：使用总变差损失
        smoothness_loss = self.total_variation_loss(x)
        
        # 范围损失：确保像素值在合理范围内
        range_loss = torch.mean(F.relu(x - 1) + F.relu(-x))
        
        # 边缘损失：使用卷积计算边缘
        edge_loss = -torch.mean(torch.abs(F.conv2d(x, 
                                                  torch.tensor([[[[1., -1.], [1., -1.]]]], 
                                                               device=self.device), 
                                                  padding=0)))
        
        return sparsity_loss + smoothness_loss + range_loss + edge_loss

    def save_batch_image(self, tensors, i, remark=None):
        """保存多张图像到一张图中
        
        Args:
            tensors (torch.Tensor): 形状为 (batch_size, 1, 28, 28) 的图像张量批次
            model_name (str): 模型名称，用于构建保存路径
            i (int): 图片的索引
        """
        # if tensors.dim() != 4 or tensors.shape[1] != 1 or tensors.shape[2:] != (28, 28):
        #     raise ValueError(f"Expected tensor shape (batch_size, 1, 28, 28), got {tensors.shape}")
        # print(tensors.size())

        # 反归一化并处理每张图片
        images = self.inverse_transform(tensors)
        images = torch.clamp(images, 0, 1)
        
        # 计算要创建的图像的行数和列数
        batch_size = images.shape[0]
        cols = int(batch_size ** 0.5)  # 假设我们想要一个接近正方形的布局
        rows = (batch_size + cols - 1) // cols  # 计算行数以容纳所有图片
        
        # 创建一个大图来保存所有图片
        plt.figure(figsize=(5 * cols, 5 * rows))
        for idx in range(batch_size):
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(images[idx].squeeze().cpu().detach().numpy(), cmap='gray')
            plt.axis('off')
        
        # 构建保存路径
        
        
        if remark is not None:
            filename = f'{self.model_name}_label_{self.target_label}_{remark}_iter_{i}.png'
        else:
            filename = f'{self.model_name}_label_{self.target_label}_iter_{i}.png'
        save_path = os.path.join(f'inversion_results/{self.model_name}', filename)
        
        # 确保保存路径存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图像
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()



    def my_attack(self, num_iterations=1000, save_interval=10):
        os.makedirs('inversion_results', exist_ok=True)
        
        # 目标标签
        iden = torch.zeros(10)
        for i in range(10):
            iden[i] = i
        iden = iden.view(-1).long().to(self.device)
        bs = iden.shape[0]
        
        lr=1e-2
        momentum=0.9
        lamda=100
        iter_times=1500
        clip_range=1
        
        dim = 32
        
        # batch_size = 64
        max_score = torch.zeros(bs)
        max_iden = torch.zeros(bs)
        z_hat = torch.zeros(bs, dim)
        
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        for random_seed in range(2):
            
            tf = time.time()
            
            torch.manual_seed(random_seed) 
            torch.cuda.manual_seed(random_seed) 
            np.random.seed(random_seed) 
            random.seed(random_seed)
            
            z = torch.randn(bs, dim).float().to(self.device)
            z.requires_grad = True
            v = torch.zeros(bs, dim).float().to(self.device)
            
            for i in range(num_iterations):
                # 生成Fake图
                fake = self.attacker(z)
                # 使用判别器生成laebl
                label = self.discriminator(fake)
                
                # temp = torch.arange(10)
                
                
                # out_test为10x10向量（10为iden长度），表明每个样本在每个标签上的分数，第n行即tensor[n]表示第n个样本在10个标签上的分数
                out_test = self.model(fake)
                # out = out_test[-1]
                
                if z.grad is not None:
                    z.grad.data.zero_()
                    
                Prior_Loss = - label.mean()
                Iden_Loss = criterion(out_test, iden)
                Total_Loss = Prior_Loss + lamda * Iden_Loss
                
                Total_Loss.backward()
                
                v_prev = v.clone()
                gradient = z.grad
                v = momentum * v - lr * gradient
                z = z + ( - momentum * v_prev + (1 + momentum) * v)
                z = torch.clamp(z.detach(), -clip_range, clip_range).float()
                z.requires_grad = True
                
                
                Prior_Loss_val = Prior_Loss.item()
                Iden_Loss_val = Iden_Loss.item()
                
                if (i+1) % save_interval == 0:
                    fake_img = self.attacker(z.detach())
                    self.save_batch_image(fake_img, i, random_seed)
                    eval_prob = self.evaluator(fake_img)
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                    print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
            
            fake = self.attacker(z)
            score = self.model(fake)
            eval_prob = self.evaluator(fake)
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
            
            cnt = 0
            for i in range(bs):
                gt = iden[i].long().item()
                if score[i, gt].item() > max_score[i].item():
                    max_score[i] = score[i, gt]
                    max_iden[i] = eval_iden[i]
                    z_hat[i, :] = z[i, :]
                if eval_iden[i].item() == gt:
                    cnt += 1
			
            interval = time.time() - tf
            print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))
            
        correct = 0
        for i in range(bs):
            gt = iden[i].item()
            if max_iden[i].item() == gt:
                correct += 1
	
        acc = correct * 1.0 / bs
        print("Acc:{:.2f}".format(acc))
            

def main():
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 训练生成对抗网络
    print("开始训练生成对抗网络...")
    # generator = train_gan(epochs=100, device=device)
    generator = Generator().to(device)
    generator.load_state_dict(torch.load('generator_epoch_2.pth', map_location=torch.device(device)))

    discriminator = Critic().to(device)
    discriminator.load_state_dict(torch.load('critic_epoch_5.pth', map_location=torch.device(device)))

    evaluator = Eval_LeNet().to(device)
    evaluator.load_state_dict(torch.load('lenet_epoch_30.pth', map_location=torch.device(device)))



    # 设置参数
    model_paths = [
        # 'lenet_epoch_1.pth',
        # 'lenet_epoch_15.pth',
        # 'lenet_epoch_30.pth',
        # 'lenet_best.pth'，
 
    ]
    
    # 对每个模型文件进行攻击
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"模型文件 {model_path} 不存在，跳过")
            continue
            
        print(f"\n{'='*50}")
        print(f"开始对模型 {model_path} 进行反转攻击...")

        # target_model = LeNet(1,10).to(device)
        # target_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # target_model.eval()

        # 每个数字类别进行攻击
        for target_label in range(1):
            attack = ModelInversionAttack(
                model_path=model_path,
                target_label=target_label,
                device=device,
                attacker=generator,  # 使用训练好的生成器
                discriminator=discriminator,  # 使用训练好的判别器
                evaluator=evaluator,  # 使用 LeNet 作为评估器
            )

            attack.my_attack(
                num_iterations=10,  # 迭代次数
                save_interval=1     # 每200次迭代保存一次图像
            )
            
            
        print(f"\n完成对模型 {model_path} 的所有标签的攻击")

if __name__ == "__main__":
    main() 