import numpy as np

State = tuple[int, int]
Action = int


class GridWorld:
    def __init__(self):
        self.reset()
        self.reward_map = np.array([[0, 0, 0, 1], [0, None, 0, -1], [0, 0, 0, 0]])
        self.actions = [0, 1, 2, 3]
        self.action_meaning = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def reset(self):
        self.state = (2, 0)

    @staticmethod
    def _next_state(s: State, a: Action) -> State:
        move_vec = {
            0: (-1, 0),  # UP
            1: (1, 0),  # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),  # RIGHT
        }
        vec = move_vec[a]
        next_s: State = (s[0] + vec[0], s[1] + vec[1])
        if GridWorld._is_in_state_range(next_s):
            return next_s
        else:
            return s

    @staticmethod
    def _is_in_state_range(s: State) -> bool:
        return 0 <= s[0] < 3 and 0 <= s[1] < 4 and s != (1, 1)

    def _reward(self, s: State, a: Action, next_s: State) -> float:
        if s == next_s:
            return 0.0
        return self.reward_map[next_s[0], next_s[1]]

    def step(self, a: Action) -> tuple[float, State, bool]:
        s = self.state
        next_s = GridWorld._next_state(s, a)
        reward = self._reward(s, a, next_s)
        done = next_s == (0, 3)
        self.state = next_s
        return reward, next_s, done

    def render(self):
        grid = np.full((3, 4), " ")
        grid[1, 1] = "X"
        grid[0, 3] = "G"
        grid[1, 3] = "B"
        grid[self.state] = "A"
        print(grid)
