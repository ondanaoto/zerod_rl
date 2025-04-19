from pydantic import BaseModel, Field


class Gene(BaseModel):
    alpha: float = Field(..., ge=0.0, le=1.0, description="Learning rate")
    epsilon: float = Field(..., ge=0.0, le=1.0, description="Exploration rate")
