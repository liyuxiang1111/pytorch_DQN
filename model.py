import torch
import torch.nn as nn
import numpy as np

# 定义标准的 DQN 网络结构
class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()
        # 卷积层 1：输入通道数为 in_channels，输出通道数为 32，卷积核大小为 8x8，步长为 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        # 卷积层 2：输入通道数为 32，输出通道数为 64，卷积核大小为 4x4，步长为 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # 卷积层 3：输入通道数为 64，输出通道数为 64，卷积核大小为 3x3，步长为 1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # 全连接层 1：输入特征数为 7*7*64（假设输入图像经过卷积和池化后的大小为 7x7x64），输出特征数为 512
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
        # 全连接层 2：输入特征数为 512，输出特征数为 num_actions，用于输出各动作的 Q 值
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        # ReLU 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x)) # 卷积层 1 + ReLU
        x = self.relu(self.conv2(x)) # 卷积层 2 + ReLU
        x = self.relu(self.conv3(x)) # 卷积层 3 + ReLU
        x = x.view(x.size(0), -1) # 展平为一维
        x = self.relu(self.fc1(x)) # 全连接层 1 + ReLU
        x = self.fc2(x) # 全连接层 2，用于输出 Q 值
        return x

# 定义对抗（Dueling）DQN 网络结构
class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        # 卷积层，与标准 DQN 网络的卷积层相同
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # 分开定义两个分支：优势函数 (Advantage) 和状态值函数 (Value)
        # 优势函数分支的全连接层 1：用于生成每个动作的优势值
        self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        # 状态值分支的全连接层 1：用于生成状态值
        self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)

        # 优势函数分支的全连接层 2：输出为 num_actions，表示每个动作的优势值
        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        # 状态值分支的全连接层 2：输出为 1，表示状态值
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        # ReLU 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        # 卷积层处理
        x = self.relu(self.conv1(x)) # 卷积层 1 + ReLU
        x = self.relu(self.conv2(x)) # 卷积层 2 + ReLU
        x = self.relu(self.conv3(x)) # 卷积层 3 + ReLU
        x = x.view(x.size(0), -1) # 展平为一维

        # 计算优势值 (Advantage)
        adv = self.relu(self.fc1_adv(x)) # 优势函数分支的全连接层 1 + ReLU
        val = self.relu(self.fc1_val(x)) # 优势函数分支的全连接层 2，用于输出每个动作的优势值

        # 计算状态值 (Value)
        adv = self.fc2_adv(adv) # 状态值分支的全连接层 1 + ReLU
        val = self.fc2_val(val).expand(x.size(0), self.num_actions) # 状态值扩展为与动作数一致的大小

        # 最终的 Q 值计算：Q(s, a) = V(s) + A(s, a) - 平均优势值
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x # 返回每个动作的 Q 值






