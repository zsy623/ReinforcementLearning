import torch
from torch import nn
from Actor import Actor
from Critic import Critic
from torch.distributions.categorical import Categorical


class A2C(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, n_step=5):
        super(A2C, self).__init__()
        self.Actor = Actor(state_size, action_size, hidden_size)
        self.Critic = Critic(state_size, hidden_size)
        self.gamma = 0.95
        self.n_step = n_step  # 添加n_step参数

    def select_action(self, state):
        x = self.Actor.forward(state)
        distribution = Categorical(logits=x)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob

    def update(self, buffer):
        if len(buffer) == 0:
            return

        # 从缓冲区提取数据
        states = torch.cat([b['state'] for b in buffer])
        actions = torch.tensor([b['action'] for b in buffer])
        rewards = torch.tensor([b['reward'] for b in buffer], dtype=torch.float32)
        next_states = torch.cat([b['next_state'] for b in buffer])
        dones = torch.tensor([b['done'] for b in buffer], dtype=torch.float32)
        log_probs = torch.stack([b['log_prob'] for b in buffer])

        # 计算n步回报
        returns = torch.zeros(len(buffer))
        # 计算最后一步的自举值
        if buffer[-1]['done']:
            G = 0.0
        else:
            G = self.Critic(buffer[-1]['next_state']).detach().squeeze().item()

        # 从后往前计算n步回报
        for i in reversed(range(len(buffer))):
            G = rewards[i] + self.gamma * G * (1 - dones[i])
            returns[i] = G

        # 计算Critic预测值
        values = self.Critic(states).squeeze()

        # 计算Critic损失（MSE）
        critic_loss = (returns - values).pow(2).mean()

        # 计算优势函数
        advantages = returns.detach() - values.detach()

        # 计算Actor损失
        actor_loss = (-log_probs * advantages).mean()

        # 清零梯度
        self.Actor.optimizer.zero_grad()
        self.Critic.optimizer.zero_grad()

        # 反向传播并更新参数
        (actor_loss + critic_loss).backward()
        self.Actor.optimizer.step()
        self.Critic.optimizer.step()
