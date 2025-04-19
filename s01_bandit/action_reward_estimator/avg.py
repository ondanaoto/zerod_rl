from .interface import ActionRewardEstimator


class AverageActionRewardEstimator(ActionRewardEstimator):
    """
    Action reward estimator that uses the average reward for each action.
    """

    def __init__(self, n_actions: int):
        if n_actions <= 0:
            raise ValueError("Number of actions must be positive")
        self.__n_actions = n_actions
        self.action_rewards: list[float] = [0.0 for _ in range(n_actions)]
        self.action_counts: list[int] = [0 for _ in range(n_actions)]

    def estimate_action_reward(self, action: int) -> float:
        return self.action_rewards[action]

    def update_estimates(self, action: int, reward: float) -> None:
        prev_reward = self.action_rewards[action]
        reward = prev_reward + (reward - prev_reward) / (self.action_counts[action] + 1)
        self.action_rewards[action] = reward
        self.action_counts[action] += 1

    @property
    def n_action(self) -> int:
        return self.__n_actions
