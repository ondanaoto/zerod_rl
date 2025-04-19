import numpy as np

from .cross import Crosser
from .environment import Environment
from .episode import Episode
from .gene import Gene
from .individual import Individual
from .select import LinearSelector
from .sort import Sorter


def main(
    # 環境の設定
    sigma: float = 0.1,
    time: int = 1000,
    arm_num: int = 10,
    # 遺伝的アルゴリズムの設定
    max_individuals: int = 200,
    cross_num: int = 30,
    mutation_rate: float = 0.3,
    # エピソード数
    epoch: int = 1000,
    topk_print: int = 5,
) -> None:
    init_rate = [np.random.rand() for _ in range(arm_num)]
    environment = Environment(init_rate=init_rate, sigma=sigma, time=time)
    crosser = Crosser(mutation_rate=mutation_rate)
    sorter = Sorter(filter_max_num=max_individuals)
    episode = Episode(
        environment=environment,
        cross_num=cross_num,
        crosser=crosser,
        selector=LinearSelector(),
        sorter=sorter,
    )

    individuals = [
        Individual(
            n_action=environment.n_action,
            gene=Gene(alpha=i / 3, epsilon=j / 3),
        )
        for i in range(3)
        for j in range(3)
    ]
    scores = [environment.eval(individual) for individual in individuals]
    individuals, scores = sorter.sort(individuals=individuals, scores=scores)
    for e in range(epoch):
        individuals, scores = episode.start(individuals=individuals, scores=scores)
        print(f"epoch: {e + 1}/{epoch}")

        avg_alpha = np.mean(
            [individual.gene.alpha for individual in individuals[:topk_print]]
        )
        avg_epsilon = np.mean(
            [individual.gene.epsilon for individual in individuals[:topk_print]]
        )
        print(f"avg_alpha: {avg_alpha}, avg_epsilon: {avg_epsilon}")
        print(f"avg_score: {np.mean(scores[:topk_print])}")


def _prompt(msg: str, cast, default):
    """空入力ならデフォルト、そうでなければ型変換して返す共通関数"""
    s = input(f"{msg} [{default}]: ").strip()
    return cast(s) if s else default


if __name__ == "__main__":
    print("=== experiment settings (press Enter to accept default) ===")
    sigma          = _prompt("variance of environment",  float, 0.1)
    time           = _prompt("time horizon per episode", int,   1000)
    arm_num        = _prompt("number of slot arms",      int,   10)
    max_individuals= _prompt("population size",          int,   200)
    cross_num      = _prompt("number of crossovers",     int,   30)
    mutation_rate  = _prompt("mutation rate",            float, 0.3)
    epoch          = _prompt("epochs (generations)",     int,   1000)

    main(
        sigma=sigma,
        time=time,
        arm_num=arm_num,
        max_individuals=max_individuals,
        cross_num=cross_num,
        mutation_rate=mutation_rate,
        epoch=epoch,
    )
