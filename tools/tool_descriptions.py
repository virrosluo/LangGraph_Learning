from typing import Literal
from pydantic import BaseModel, Field

class routeResponse(BaseModel):
    next: Literal["Researcher", "Coder", "FINISH"] = Field(description="Choosing the next worker to act")