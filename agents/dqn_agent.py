# agents/dqn_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Embedded ReplayBuffer (no external import needed)
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.stack(s), np.array(a), np.array(r), np.stack(ns), np.array(d))

    def __len__(self):
        return len(self.buffer)

# Keep the model import as-is (this is part of your project structure)
from models.dqn_network import DQNNetwork

class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        eps_decay=0.995,
        eps_min=0.01,
        buffer_size=100_000,
        batch_size=64,
        target_update=1000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = DQNNetwork(state_size, action_size).to(self.device)
        self.target = DQNNetwork(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def act(self, state, valid_actions=None):
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions if valid_actions is not None else range(self.action_size)))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy(s)[0]
            if valid_actions is not None:
                mask = torch.full((self.action_size,), float("-inf"), device=self.device)
                mask[valid_actions] = 0.0
                q = q + mask
            return int(torch.argmax(q).item())

    def remember(self, s, a, r, ns, d):
        self.buffer.push(s, a, r, ns, d)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        q_pred = self.policy(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            q_next = self.target(ns).max(1)[0]
            q_target = r + (1.0 - d) * self.gamma * q_next

        loss = self.loss_fn(q_pred, q_target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())

        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

        return float(loss.item())

    def save(self, path):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "target": self.target.state_dict(),
                "optim": self.optim.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.target.load_state_dict(ckpt["target"])
        self.optim.load_state_dict(ckpt["optim"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.steps = ckpt.get("steps", 0)
