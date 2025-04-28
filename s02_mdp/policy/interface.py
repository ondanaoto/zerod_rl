from abc import ABC, abstractmethod

from s02_mdp.action import Action
from s02_mdp.state import State


class Policy(ABC):
    @abstractmethod
    def get(self, state: State) -> dict[Action, float]: ...
