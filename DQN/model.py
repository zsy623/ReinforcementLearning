import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ReplayBuffer

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN_Agent:
    def __init__(self, state_size, action_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_size = state_size
        self.action_size = action_size

        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()

        self.gamma = 0.99
        self.buffer = ReplayBuffer.ReplayBuffer(10000)
        self.epsilon = 0.90
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.update_frequency = 100

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = torch.argmax(q_values).item()
            return action

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        left_q_value = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_value = self.target_net(next_states).gather(1, actions)
            right_q_value = rewards + (1 - dones.float()) * self.gamma * next_q_value

        loss = self.loss(left_q_value, right_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
