from enum import Enum

from pydantic import BaseModel, Field


class Masu(Enum):
    L1 = "L1"
    L2 = "L2"


class State(BaseModel):
    position: Masu = Field(default=Masu.L1)
