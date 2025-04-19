from .interface import ActionRewardEstimator


class ExponentialActionRewardEstimator(ActionRewardEstimator):
    def __init__(self, n_action: int, alpha: float):
        if n_action <= 0:
            raise ValueError("n_action must be positive. n_action: ", n_action)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must in range from 0.0 to 1.0. alpha: ", alpha)
        self.__n_action = n_action
        self.__alpha = alpha
        self.__action_values = [0.0 for _ in range(n_action)]

    def estimate_action_reward(self, action: int) -> float:
        return self.__action_values[action]

    def update_estimates(self, action: int, reward: float) -> None:
        action_value = self.__action_values[action]
        updated_action_value = action_value + self.__alpha * (reward - action_value)
        self.__action_values[action] = updated_action_value

    @property
    def n_action(self) -> int:
        return self.__n_action
