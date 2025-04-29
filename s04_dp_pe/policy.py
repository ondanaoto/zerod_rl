from c01_grid_world.env import Action, State


class Policy:
    def __init__(self):
        self._prob_dict = {
            state: {action: 1.0 / len(Action) for action in Action}
            for state in State.range()
        }

    def __call__(self, state: State) -> dict[Action, float]:
        """
        Returns a dictionary of actions and their probabilities.
        """
        return self._prob_dict[state]

    def update_greedy(self, d: dict[State, Action]) -> None:
        for state, action in d.items():
            for a in Action:
                prob = 1.0 if a == action else 0.0
                self._prob_dict[state][a] = prob
