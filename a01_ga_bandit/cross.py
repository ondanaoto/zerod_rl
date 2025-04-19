import numpy as np
from pydantic import BaseModel, Field

from .gene import Gene


class Crosser(BaseModel):
    mutation_rate: float = Field(ge=0.0, le=1.0)

    def cross(self, parent1: Gene, parent2: Gene) -> tuple[Gene, Gene]:
        """
        Crossover two parents to create two children.
        """
        child1 = Gene(alpha=parent1.alpha, epsilon=parent2.epsilon)
        child2 = Gene(alpha=parent2.alpha, epsilon=parent1.epsilon)
        if np.random.rand() < self.mutation_rate:
            self._mutate(child1)
            self._mutate(child2)
        return child1, child2

    def _mutate(self, gene: Gene) -> None:
        if np.random.rand() < 0.5:
            self._mutate_alpha(gene)
        else:
            self._mutate_epsilon(gene)

    @staticmethod
    def _mutate_alpha(gene: Gene) -> None:
        gene.alpha = np.random.random()

    @staticmethod
    def _mutate_epsilon(gene: Gene) -> None:
        gene.epsilon = np.random.random()
