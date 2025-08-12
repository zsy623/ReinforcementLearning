import torch
from torch import nn
from Actor import Actor
from Critic import Critic
from torch.distributions.categorical import Categorical


class A2C(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(A2C, self).__init__()
        self.Actor = Actor(state_size, action_size, hidden_size)
        self.Critic = Critic(state_size, hidden_size)
        # 移除 log_probs 和 rewards 列表，因为我们采用的是单步更新
        self.gamma = 0.95

    def select_action(self, state):
        x = self.Actor.forward(state)
        distribution = Categorical(logits=x)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob

    def update(self, state, action, reward, next_state, done, log_prob):
        # 确保输入是正确的 PyTorch Tensor 格式
        # state 在 train.py 中已经被处理，所以这里不需要再处理
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        action_tensor = torch.tensor([[action]]).long()
        done_tensor = torch.tensor(done).float()

        # 计算 TD 目标值和 TD 误差 (优势)
        with torch.no_grad():
            next_value = self.Critic(next_state)
            td_target = reward + self.gamma * next_value * (1 - done_tensor)

        predicted_value = self.Critic(state)
        td_error = td_target - predicted_value

        # Critic 损失：TD 误差的均方误差
        critic_loss = td_error.pow(2).mean()

        # Actor 损失：策略梯度，使用优势函数进行加权
        actor_loss = -log_prob * td_error.detach()

        # 清零梯度
        self.Actor.optimizer.zero_grad()
        self.Critic.optimizer.zero_grad()

        # 反向传播并更新参数
        actor_loss.backward()
        critic_loss.backward()

        self.Actor.optimizer.step()
        self.Critic.optimizer.step()
