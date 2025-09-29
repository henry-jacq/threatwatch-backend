from sqlalchemy import Column, Integer, String, Boolean, DateTime
from app.db.session import Base
import datetime

class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    ip = Column(String, index=True, unique=True)
    tags = Column(String, nullable=True)
    online = Column(Boolean, default=False)
    last_seen = Column(DateTime, default=datetime.datetime.utcnow)
