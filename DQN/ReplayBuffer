import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 双端队列实现有限缓冲区

    def push(self, state, action, reward, next_state, done):
        """存储转移样本"""
        state = np.array(state, dtype=np.float32)  # 转为numpy数组
        next_state = np.array(next_state, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))  # 添加新样本

    def sample(self, batch_size):
        """随机采样小批量样本"""
        # 随机选择batch_size个样本
        batch = random.sample(self.buffer, batch_size)
        # 解压样本并转换为PyTorch张量
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),  # 状态张量
            torch.LongTensor(actions).unsqueeze(1),  # 动作张量（增加维度）
            torch.FloatTensor(rewards).unsqueeze(1),  # 奖励张量
            torch.FloatTensor(next_states),  # 下一状态张量
            torch.BoolTensor(dones).unsqueeze(1)  # 终止标志
        )

    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)
