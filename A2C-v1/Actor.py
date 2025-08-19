import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # 添加层归一化
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)  # 添加层归一化
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)  # 添加权重衰减

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        # 使用正交初始化
        init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        init.constant_(self.fc1.bias, 0.0)
        init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        init.constant_(self.fc2.bias, 0.0)
        # 最后一层使用较小的权重初始化
        init.orthogonal_(self.fc3.weight, gain=0.01)
        init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))  # 层归一化在激活函数前
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x