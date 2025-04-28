from s02_mdp.action import Action
from s02_mdp.state import State

from .interface import Policy


class RandomPolicy(Policy):
    """
    A policy that selects actions uniformly at random.
    """

    def get(self, state: State) -> dict[Action, float]:
        """
        Returns a dictionary of actions and their probabilities.
        """
        return dict.fromkeys(Action, 0.5)
