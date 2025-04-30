from collections import defaultdict
from c01_grid_world.env import Action, GridWorld, State

from .policy import Policy


class ValueFunction:
    def __init__(self, policy: Policy, gamma: float) -> None:
        self._value_dict: dict[State, float] = defaultdict(lambda: 0.0)
        self.policy = policy
        self.__gamma = gamma

    def __call__(self, state: State) -> float:
        """
        Returns the value of the state.
        """
        return self._value_dict[state]

    def update(self, policy: Policy | None = None) -> float:
        """
        Updates the value function based on the current policy.
        """
        env = GridWorld()
        old_value_dict = self._value_dict.copy()
        if policy is not None:
            self.policy = policy

        for state in State.range():
            action_probs = self.policy(state)
            next_value = 0.0
            if state.row == 0 and state.col == 3:
                continue
            for action, prob in action_probs.items():
                env.reset(state)
                next_state, reward, _ = env.step(action)
                next_value += prob * (
                    reward + self.__gamma * old_value_dict[next_state]
                )
            self._value_dict[state] = next_value

        # Check for convergence
        max_diff = max(
            abs(old_value_dict[state] - self._value_dict[state])
            for state in State.range()
        )
        return max_diff

    def get_greedy_policy(self) -> Policy:
        """
        Returns the greedy policy based on the current value function.
        """
        env = GridWorld()
        greedy_policy: dict[State, Action] = {}
        for state in State.range():
            max_value = float("-inf")
            best_action: Action
            for action in Action:
                env.reset(state)
                next_state, reward, _ = env.step(action)
                action_value = reward + self.__gamma * self._value_dict[next_state]
                if action_value > max_value:
                    max_value = action_value
                    best_action = action
            greedy_policy[state] = best_action
        return Policy.from_greedy(greedy_policy)

    def render(self) -> None:
        """
        Renders the value function.
        """
        value_grid: list[list[str | None]] = [
            [None for _ in range(4)] for _ in range(3)
        ]
        for state, value in self._value_dict.items():
            value_grid[state.row][state.col] = f"{value:+.2f}"
        print("Value Function:")
        for row in value_grid:
            print(" ".join(str(v) if v is not None else "     " for v in row))
