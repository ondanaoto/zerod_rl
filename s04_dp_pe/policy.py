from collections import defaultdict

from c01_grid_world.env import Action, State


class Policy:
    def __init__(
        self, prob_dict: dict[State, dict[Action, float]] | None = None
    ) -> None:
        if prob_dict is None:
            prob_dict = defaultdict(
                lambda: {action: 1.0 / len(Action) for action in Action}
            )
        self._prob_dict = prob_dict

    def __call__(self, state: State) -> dict[Action, float]:
        """
        Returns a dictionary of actions and their probabilities.
        """
        return self._prob_dict[state]

    @classmethod
    def from_greedy(cls, d: dict[State, Action], epsilon: float = 0.0) -> "Policy":
        prob_dict: dict[State, dict[Action, float]] = defaultdict(
            lambda: dict.fromkeys(Action, 0.0)
        )
        for state, action in d.items():
            for a in Action:
                prob = (
                    1.0 - epsilon * (1 - 1 / len(Action))
                    if a == action
                    else epsilon / len(Action)
                )
                prob_dict[state][a] = prob
        return cls(prob_dict)
