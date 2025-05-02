from collections import defaultdict

import numpy as np

from c01_grid_world.env import Action, GridWorld, State
from s04_dp.policy import Policy

from .history import History


class MonteCarloOptimizer:
    def __init__(
        self, gamma: float = 0.9, alpha: float = 0.1, epsilon: float = 0.1
    ) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.pi = Policy()
        self.b = Policy()
        self.__qa_values: dict[tuple[State, Action], float] = defaultdict(lambda: 0.0)

    def _mc_start(self) -> History:
        env = GridWorld()
        history = History()
        done = False
        while not done:
            state = env.state
            action_prob = self.b(state)
            actions: list[Action] = list(action_prob.keys())
            probs: list[float] = list(action_prob.values())
            action_idx: int = np.random.choice(len(actions), p=probs)
            action = actions[action_idx]
            _, reward, done = env.step(action)
            history.add(state, action, reward)

        return history

    def update_policy(self, episode_num: int = 1000) -> None:
        """valueの更新が終わるまでself.piとself.bを更新する"""
        for _ in range(episode_num):
            history = self._mc_start()
            revenue: float = 0
            rho = 1.0
            for state, action, reward in reversed(history.get()):
                q_value = self.__qa_values[state, action]
                revenue = reward + self.gamma * rho * revenue
                self.__qa_values[state, action] += self.alpha * (revenue - q_value)
                rho *= self.pi(state)[action] / self.b(state)[action]

            greedy_dict: dict[State, Action] = defaultdict(lambda: Action.UP)
            for state in State.range():
                greedy_action: Action
                greatest_value: float = float("-inf")
                for action in Action:
                    val = self.__qa_values[state, action]
                    if val > greatest_value:
                        greedy_action = action
                        greatest_value = val
                greedy_dict[state] = greedy_action
            self.pi = Policy.from_greedy(greedy_dict)
            self.b = Policy.from_greedy(greedy_dict, self.epsilon)

    def render_pi(self) -> None:
        value_grid: list[list[Action | None]] = [
            [None for _ in range(4)] for _ in range(3)
        ]
        for state in State.range():
            action_dict = self.pi(state)
            actions = list(action_dict.keys())
            probs = list(action_dict.values())
            action = actions[np.argmax(probs)]
            value_grid[state.row][state.col] = action

        print("Best Actions:")
        for row in value_grid:
            print(" ".join(str(v) if v is not None else "     " for v in row))
