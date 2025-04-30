from c01_grid_world.env import Action, State


class History:
    def __init__(self) -> None:
        self.__history: list[tuple[State, Action, float]] = []

    def add(self, state: State, action: Action, reward: float):
        item = (state, action, reward)
        self.__history.append(item)

    def get(self) -> list[tuple[State, Action, float]]:
        return self.__history
