from collections import defaultdict

import numpy as np

from c01_grid_world.env import Action, State


class QLearningAgent:
    def __init__(
        self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1
    ) -> None:
        self.Q: dict[tuple[State, Action], float] = defaultdict(lambda: 0.0)
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon

    def update(
        self, state: State, action: Action, reward: float, next_state: State
    ) -> None:
        self.Q[state, action] += self.alpha * (
            reward
            + self.gamma * (max(self.Q[next_state, a] for a in Action))
            - self.Q[state, action]
        )

    def b(self, state: State, epsilon: float | None = None) -> dict[Action, float]:
        action_idx = np.argmax([self.Q[state, a] for a in Action])
        if epsilon is None:
            epsilon = self.epsilon
        actions = list(Action)
        best_action = actions[action_idx]
        action_prob = {a: epsilon / len(actions) for a in actions}
        action_prob[best_action] += 1 - epsilon
        return action_prob

    def get_action(self, state: State, epsilon: float | None = None) -> Action:
        action_prob = self.b(state, epsilon)
        actions = list(action_prob.keys())
        probs = list(action_prob.values())
        action_idx = np.random.choice(len(actions), p=probs)
        return actions[action_idx]

    def render_pi(self) -> None:
        value_grid: list[list[Action | None]] = [
            [None for _ in range(4)] for _ in range(3)
        ]
        for state in State.range():
            action_dict = self.b(state, epsilon=0.0)
            actions = list(action_dict.keys())
            probs = list(action_dict.values())
            action = actions[np.argmax(probs)]
            value_grid[state.row][state.col] = action

        print("Best Actions:")
        for row in value_grid:
            print(" ".join(str(v) if v is not None else "     " for v in row))
