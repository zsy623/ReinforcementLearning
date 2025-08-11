import torch
import torch.nn as nn
import torch.optim as optim
from PolicyNetwork import Policy
from torch.distributions import Categorical

class REINFORCE(nn.Module):
    def __init__(self, state_size, action_size):
        super(REINFORCE, self).__init__()
        self.policy_network = Policy(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(),lr=1e-3)
        self.gamma = 0.99
        self.save_log_prob = []
        self.rewards = []

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        self.save_log_prob.append(m.log_prob(action))
        return action.item()

    def update(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        policy_loss = []
        for log_prob, r in zip(self.save_log_prob, returns):
            policy_loss.append(-log_prob * r)

        self.optimizer.zero_grad()
        policy_loss = sum(policy_loss)
        policy_loss.backward()
        self.optimizer.step()

        self.save_log_prob = []
        self.rewards = []





