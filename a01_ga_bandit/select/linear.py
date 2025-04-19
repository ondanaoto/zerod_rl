import numpy as np

from .interface import Individual, ParentSelector


class LinearSelector(ParentSelector):
    @staticmethod
    def select_parents(
        individuals: list[Individual], scores: list[float]
    ) -> tuple[Individual, Individual]:
        """
        Select two parents based on a linear combination of their scores.
        """
        probabilities = np.array(scores) / sum(scores)
        parent_indices = np.random.choice(len(individuals), size=2, p=probabilities)
        return individuals[parent_indices[0]], individuals[parent_indices[1]]
