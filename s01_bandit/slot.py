from pydantic import BaseModel, Field


class Slot(BaseModel):
    rate: float = Field(..., ge=0, le=1)

    def play(self) -> int:
        """
        Simulate playing the slot machine.
        Returns 1 for win, 0 for loss.
        """
        from random import random

        return 1 if random() < self.rate else 0
