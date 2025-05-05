import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from c01_grid_world.env import Action, State


def one_hot(state: State) -> torch.Tensor:
    idx = state.row * 4 + state.col
    encoding = torch.zeros((1, 12), dtype=torch.float32)
    encoding[0, idx] = 1.0
    return encoding


class QNet(nn.Module):
    def __init__(self, hidden_dim: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(12, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepQLearningAgent:
    def __init__(
        self,
        hidden_dim: int = 100,
        lr: float = 1e-4,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ) -> None:
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.QNet = QNet(hidden_dim)
        self.optimizer = optim.AdamW(self.QNet.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def update(
        self, state: State, action: Action, reward: float, next_state: State, done: bool
    ) -> None:
        with torch.no_grad():
            target = (
                reward
                + self.gamma
                * (1 - int(done))
                * torch.max(self.QNet(one_hot(next_state)), dim=1).values
            )
        current = self.QNet(one_hot(state))[:, action.value]
        loss = self.criterion(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state: State) -> Action:
        if np.random.rand() < self.epsilon:
            action_idx = np.random.choice(4)
            return list(Action)[action_idx]
        action_tensor: torch.Tensor = self.QNet(one_hot(state))
        idx: int = int(torch.argmax(action_tensor).item())
        actions = list(Action)
        best_action = actions[idx]
        return best_action

    def render_pi(self) -> None:
        value_grid: list[list[Action | None]] = [
            [None for _ in range(4)] for _ in range(3)
        ]
        for state in State.range():
            action = self.get_action(state)
            value_grid[state.row][state.col] = action

        print("Best Actions:")
        for row in value_grid:
            print(" ".join(str(v) if v is not None else "     " for v in row))
