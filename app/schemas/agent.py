from pydantic import BaseModel
from datetime import datetime

class AgentBase(BaseModel):
    name: str
    ip: str
    tags: str | None = None

class AgentCreate(AgentBase):
    pass

class AgentOut(AgentBase):
    id: int
    online: bool
    last_seen: datetime

    model_config = {"from_attributes": True}