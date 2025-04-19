from abc import ABC, abstractmethod

from ..individual import Individual


class ParentSelector(ABC):
    """
    Abstract base class for action selectors.
    """

    @abstractmethod
    def select_parents(
        self, individuals: list[Individual], scores: list[float]
    ) -> tuple[Individual, Individual]:
        """
        Select an action based on the provided action values.
        """
        ...
