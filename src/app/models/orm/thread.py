from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from .base import Base

if TYPE_CHECKING:
    from .user import User
    
class Thread(Base):
    __tablename__ = 'threads'

    thread_id : Mapped[uuid.UUID] = mapped_column(primary_key = True , default = uuid.uuid4, index = True)
    title : Mapped[str] = mapped_column(String(150) , default = 'old chat session',index = True)
    owner_id : Mapped[uuid.UUID] = mapped_column(ForeignKey('users.user_id', ondelete='CASCADE')) 
    created_at : Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda:datetime.now(UTC)
    )
    # realtionship
    creator : Mapped[User] = relationship(back_populates='owned_threads' , lazy = 'noload')
