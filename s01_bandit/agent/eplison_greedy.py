import numpy as np

from s01_bandit.action_value_estimator import ActionRewardEstimator

from .interface import Agent


class EpsilonGreedyAgent(Agent):
    def __init__(self, estimator: ActionRewardEstimator, epsilon: float):
        self.estimator = estimator
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError(
                "epsilon must be in range from 0.0 to 1.0. epsilon: ", epsilon
            )
        self.epsilon = epsilon

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return self._explore()
        return self._exploit()

    def _exploit(self) -> int:
        action_rewards = [
            self.estimator.estimate_action_reward(action)
            for action in range(self.estimator.n_action)
        ]
        return np.argmax(action_rewards).item()

    def _explore(self) -> int:
        return np.random.randint(self.estimator.n_action)
