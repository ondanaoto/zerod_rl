from typing import Any

from pydantic import BaseModel, PrivateAttr, field_validator

from s01_bandit.slot import Slot

from .interface import Bandit


class FixBandit(BaseModel, Bandit):
    rates: list[float]
    _slots: list[Slot] = PrivateAttr()

    @field_validator("rates")
    def validate_rates(cls, rates: list[float]) -> list[float]:
        if len(rates) == 0:
            raise ValueError("Rates list cannot be empty")
        if not all(0 <= rate <= 1 for rate in rates):
            raise ValueError("All rates must be between 0 and 1")
        return rates

    def model_post_init(self, __context: Any) -> None:
        self._slots: list[Slot] = [Slot(rate=rate) for rate in self.rates]

    def play(self, arm: int) -> int:
        if arm < 0 or arm >= len(self._slots):
            raise ValueError("Invalid arm number. arm: ", arm)
        return self._slots[arm].play()

    @property
    def arm_count(self) -> int:
        return len(self._slots)
