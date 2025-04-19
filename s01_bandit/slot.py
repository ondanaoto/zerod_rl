from random import random

from pydantic import BaseModel, Field


class Slot(BaseModel):
    rate: float = Field(..., ge=0, le=1, description="Win rate of the slot machine")

    def play(self) -> int:
        """
        Simulate playing the slot machine.
        Returns 1 for win, 0 for loss.
        """

        return 1 if random() < self.rate else 0
