from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class State(BaseModel):
    row: int = Field(..., ge=0, le=2)
    col: int = Field(..., ge=0, le=3)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_in_range(self) -> "State":
        if not self.is_in_range(self.row, self.col):
            raise ValueError("State (1, 1) is not allowed", self.row, self.col)
        return self

    @staticmethod
    def is_in_range(x: int, y: int) -> bool:
        return 0 <= x < 3 and 0 <= y < 4 and (x, y) != (1, 1)

    @classmethod
    def range(self) -> list["State"]:
        return [
            State(row=x, col=y)
            for x in range(3)
            for y in range(4)
            if State.is_in_range(x, y)
        ]


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @property
    def vec(self) -> tuple[int, int]:
        return {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
        }[self]


class GridWorld:
    def __init__(self, s: State | None = None):
        self.reset(s)
        self.__REWARD_MAP = np.array([[0, 0, 0, 1], [0, None, 0, -1], [0, 0, 0, 0]])

    def reset(self, s: State | None = None) -> None:
        if s is None:
            s = State(row=2, col=0)
        self.state = s

    @staticmethod
    def _next_state(s: State, a: Action) -> State:
        try:
            next_s = State(row=s.row + a.vec[0], col=s.col + a.vec[1])
        except ValueError:
            next_s = s
        return next_s

    def _reward(self, s: State, a: Action, next_s: State) -> float:
        if s == next_s:
            return 0.0
        return self.__REWARD_MAP[next_s.row, next_s.col]

    def step(self, a: Action) -> tuple[State, float, bool]:
        s = self.state
        next_s = GridWorld._next_state(s, a)
        reward = self._reward(s, a, next_s)
        done = next_s.row == 0 and next_s.col == 3
        self.state = next_s
        return next_s, reward, done

    def render(self):
        grid = np.full((3, 4), " ")
        grid[1, 1] = "X"
        grid[0, 3] = "G"
        grid[1, 3] = "B"
        grid[self.state.row, self.state.col] = "A"
        print(grid)
