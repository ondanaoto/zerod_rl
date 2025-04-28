from s02_mdp.action import Action
from s02_mdp.state import Masu, State

from .interface import Policy


class BestPolicy(Policy):
    def get(self, state: State) -> dict[Action, float]:
        """
        Returns a dictionary of actions and their probabilities.
        """
        if state.position == Masu.L1:
            return {Action.LEFT: 0.0, Action.RIGHT: 1.0}
        elif state.position == Masu.L2:
            return {Action.LEFT: 1.0, Action.RIGHT: 0.0}
        else:
            raise ValueError("Invalid state")
