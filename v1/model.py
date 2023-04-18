"""
define DQN network and DQN anget
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class DQN(nn.Module):
    """
    input dim: [..., state_dim]
    output dim: [..., action_dim]
    """
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state = state.unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, memory, batch_size):
        # 只有记忆池大于batch_size的时候，才进行经验抽取和网络参数更新
        if len(memory) < batch_size:
            return

        # 经验抽取
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 对其中的各个参数进行变换
        states = torch.tensor(states, dtype=torch.float32).to(self.device).unsqueeze(0)  # [1, batch_size]
        states = states.permute(1, 0)  # [batch_size, 1]
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # 计算当前状态下各个动作的Q值
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states.unsqueeze(-1)).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 更新网络参数
        loss = self.loss_fn(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

