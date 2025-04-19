from abc import ABC, abstractmethod


class ActionRewardEstimator(ABC):
    """
    Abstract base class for action reward estimators.
    """

    @abstractmethod
    def estimate_action_reward(self, action: int) -> float:
        """
        Estimate the reward for a given action.
        """
        ...

    @abstractmethod
    def update_estimates(self, action: int, reward: float) -> None:
        """
        Update the action reward estimates based on the observed reward.
        """
        ...

    @property
    @abstractmethod
    def n_action(self) -> int: ...
