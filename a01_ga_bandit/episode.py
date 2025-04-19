from .cross import Crosser
from .environment import Environment
from .individual import Individual
from .select import ParentSelector
from .sort import Sorter


class Episode:
    def __init__(
        self,
        environment: Environment,
        crosser: Crosser,
        selector: ParentSelector,
        cross_num: int,
        sorter: Sorter,
    ):
        self.environment = environment
        self.crosser = crosser
        self.selector = selector
        self.sorter = sorter
        self.cross = cross_num

    def start(
        self, individuals: list[Individual], scores: list[float]
    ) -> tuple[list[Individual], list[float]]:
        # select individuals
        children = []
        for _ in range(self.cross):
            parent1, parent2 = self.selector.select_parents(
                individuals=individuals, scores=scores
            )
            child1_gene, child2_gene = self.crosser.cross(parent1.gene, parent2.gene)
            children.append(
                Individual(n_action=self.environment.n_action, gene=child1_gene)
            )
            children.append(
                Individual(n_action=self.environment.n_action, gene=child2_gene)
            )
        self.environment.reset()
        child_scores = [self.environment.eval(individual=child) for child in children]
        individuals.extend(children)
        scores.extend(child_scores)

        return self.sorter.sort(individuals=individuals, scores=scores)
