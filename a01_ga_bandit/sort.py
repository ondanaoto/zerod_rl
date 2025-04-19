import numpy as np

from .individual import Individual


class Sorter:
    def __init__(self, filter_max_num: int | None = None):
        self.filter_max_num = filter_max_num

    def sort(
        self, individuals: list[Individual], scores: list[float]
    ) -> tuple[list[Individual], list[float]]:
        """
        Sort individuals and scores in descending order based on scores.
        """
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = [scores[i] for i in sorted_indices]
        sorted_individuals: list[Individual] = [individuals[i] for i in sorted_indices]
        if self.filter_max_num is not None:
            sorted_individuals = sorted_individuals[: self.filter_max_num]
            sorted_scores = sorted_scores[: self.filter_max_num]
        return sorted_individuals, sorted_scores
