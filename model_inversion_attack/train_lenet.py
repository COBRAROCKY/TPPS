import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
import math
from typing import List
import torch.nn as nn
import torch.nn.functional as F


# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    # def __init__(self, input_channels, output_channels):
    #     super(LeNet, self).__init__()
    #     self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
    #     self.conv2 = nn.Conv2d(10, 20, kernel_size= 5)
    #     self.conv2_drop = nn.Dropout2d()
    #     self.fc1 = nn.Linear(320,50)
    #     self.fc2 = nn.Linear(50, output_channels)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = F.max_pool2d(x, 2)
    #     x = F.relu(x)
    #     x = self.conv2(x)
    #     x = self.conv2_drop(x)
    #     x = F.max_pool2d(x,2)
    #     x = F.relu(x)
    #     x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     return  x



# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST训练和测试数据集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 随机选择1/60的训练数据
total_samples = len(train_dataset)
subset_size = total_samples // 2
indices = random.sample(range(total_samples), subset_size)
subset_dataset = Subset(train_dataset, indices)

# 创建数据加载器
train_loader = DataLoader(subset_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 初始化模型
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 添加模型裁剪函数
def clip_parameters(model, threshold):
    """对模型参数进行阈值裁剪"""
    norm = torch.norm(torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]))
    if norm > threshold:
        scale = threshold / norm
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(scale)

# 添加高斯噪声函数
def add_gaussian_noise(model, noise_scale):
    """添加高斯噪声实现差分隐私"""
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_scale
            param.add_(noise)

# 添加测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)')
    return accuracy

def compute_noise_multiplier(epsilon: float, delta: float, sample_rate: float, epochs: int) -> float:
    """
    计算差分隐私高斯机制所需的噪声乘数
    
    参数:
    epsilon (float): 隐私预算 ε
    delta (float): 隐私松弛参数 δ
    sample_rate (float): 批次大小与总数据集大小的比率
    epochs (int): 训练轮数
    
    返回:
    float: 噪声乘数 σ
    """
    c = math.sqrt(2 * math.log(1.25 / delta))
    return c * math.sqrt(epochs * sample_rate) / epsilon

def get_noise_scale(epsilon: float, delta: float, batch_size: int, dataset_size: int, 
                    clip_threshold: float, epochs: int) -> float:
    """
    计算差分隐私所需的噪声规模
    
    参数:
    epsilon (float): 隐私预算
    delta (float): 隐私松弛参数
    batch_size (int): 批次大小
    dataset_size (int): 数据集大小
    clip_threshold (float): 梯度裁剪阈值
    epochs (int): 训练轮数
    
    返回:
    float: 噪声规模
    """
    sample_rate = batch_size / dataset_size
    noise_multiplier = compute_noise_multiplier(epsilon, delta, sample_rate, epochs)
    return clip_threshold * noise_multiplier

# 修改训练函数，添加差分隐私开关
def train(epochs: int, use_dp: bool = True, epsilon: float = 10.0, 
          delta: float = 1e-5, clip_threshold: float = 1.0):
    """
    训练函数
    
    参数:
    epochs (int): 训练轮数
    use_dp (bool): 是否使用差分隐私
    epsilon (float): 隐私预算
    delta (float): 隐私松弛参数
    clip_threshold (float): 梯度裁剪阈值
    """
    noise_scale = 0
    if use_dp:
        # 计算噪声规模
        noise_scale = get_noise_scale(
            epsilon=epsilon,
            delta=delta,
            batch_size=train_loader.batch_size,
            dataset_size=len(subset_dataset),
            clip_threshold=clip_threshold,
            epochs=epochs
        )
        print(f"启用差分隐私保护")
        print(f"Privacy parameters: ε={epsilon}, δ={delta}")
        print(f"Computed noise scale: {noise_scale}")
    else:
        print("未启用差分隐私保护")
    
    best_accuracy = 0
    accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 如果启用差分隐私，则应用梯度裁剪
            if use_dp:
                clip_parameters(model, clip_threshold)
            
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # 如果启用差分隐私，在每个epoch结束后添加高斯噪声
        if use_dp:
            add_gaussian_noise(model, noise_scale)
        
        # 测试阶段
        print(f'\nEvaluating model after epoch {epoch+1}:')
        current_accuracy = test(model, device, test_loader)
        accuracies.append(current_accuracy)
        
        # 保存最佳模型
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), 'lenet_best.pth')
            print(f'New best model saved with accuracy: {best_accuracy:.2f}%')
        
        # 在指定轮次保存模型
        if epoch + 1 in [1, 15, 30]:
            torch.save(model.state_dict(), f'lenet_epoch_{epoch+1}.pth')
            print(f'Model saved at epoch {epoch+1}')
        
        if use_dp:
            print(f'Epoch {epoch+1}: Applied clipping threshold {clip_threshold} '
                  f'and noise scale {noise_scale}')
        print('-' * 60)
    
    # 打印训练总结
    print('\nTraining Summary:')
    print(f'Best accuracy: {best_accuracy:.2f}%')
    print(f'Accuracy progression: {[f"{acc:.2f}%" for acc in accuracies]}')
    if use_dp:
        print(f'Final privacy guarantee: (ε={epsilon}, δ={delta})')
    else:
        print('No privacy guarantees (DP was disabled)')

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 主程序
if __name__ == "__main__":
    set_seed()
    
    # 设置训练参数
    USE_DP = False      # 差分隐私开关
    EPSILON = 10.0     # 隐私预算
    DELTA = 1e-5       # 隐私松弛参数
    CLIP_THRESHOLD = 1.0  # 梯度裁剪阈值
    
    # 开始训练
    train(
        epochs=30,
        use_dp=USE_DP,
        epsilon=EPSILON,
        delta=DELTA,
        clip_threshold=CLIP_THRESHOLD
    ) 