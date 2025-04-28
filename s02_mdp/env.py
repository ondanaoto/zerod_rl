from .action import Action
from .state import Masu, State


class Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = State()

    @staticmethod
    def action_reward(state: State, action: Action):
        if action == Action.LEFT:
            if state.position == Masu.L1:
                return -1
            elif state.position == Masu.L2:
                return 1
            else:
                raise ValueError("Invalid state")
        elif action == Action.RIGHT:
            if state.position == Masu.L2:
                return -1
            elif state.position == Masu.L1:
                return 0
        raise ValueError("Invalid action")

    @staticmethod
    def next_state(s: State, a: Action) -> State:
        if s.position == Masu.L1:
            if a == Action.RIGHT:
                s.position = Masu.L2
            elif a == Action.LEFT:
                pass
            return s
        elif s.position == Masu.L2:
            if a == Action.RIGHT:
                pass
            elif a == Action.LEFT:
                s.position = Masu.L1
            return s

    def step(self, a: Action) -> tuple[State, int]:
        """
        Takes an action and returns the next state and reward.
        """
        reward = self.action_reward(self.state, a)
        self.state = self.next_state(self.state, a)
        return self.state, reward
