import random

from s01_bandit.slot import Slot

from .interface import Bandit


class NonStatBandit(Bandit):
    def __init__(self, init_rate: list[float], sigma: float):
        if len(init_rate) <= 0:
            raise ValueError("init_rate must be nonempty. init_rate: ", init_rate)
        if any(rate < 0 or rate > 1 for rate in init_rate):
            raise ValueError(
                "each rate must be in range from 0.0 to 1.0. init_rate: ", init_rate
            )
        if sigma <= 0.0:
            raise ValueError("sigma must be positive. sigma: ", sigma)
        self.slots: list[Slot] = [Slot(rate=rate) for rate in init_rate]
        self.__arm_count = len(init_rate)
        self.__sigma = sigma

    def play(self, arm: int) -> int:
        slot = self.slots[arm]
        reward = self.slots[arm].play()
        # update_rate
        new_rate = slot.rate + self.__sigma * random.gauss()
        if new_rate < 0.0:
            new_rate = 0.0
        if new_rate > 1.0:
            new_rate = 1.0
        self.slots[arm].rate = new_rate

        return reward

    @property
    def arm_count(self) -> int:
        return self.__arm_count
