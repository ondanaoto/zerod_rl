from abc import ABC, abstractmethod


class Bandit(ABC):
    @abstractmethod
    def play(self, arm: int) -> int: ...

    @property
    @abstractmethod
    def arm_count(self) -> int: ...
