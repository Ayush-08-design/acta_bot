from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    UUID,
    Enum,
    String,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from .base import Base

if TYPE_CHECKING: # prevents circular imports
    from .thread import Thread
    
class UserRole(str , enum.Enum):
    ADMIN = 'admin'
    MANAGER = 'manager'
    USER = 'user'
    
class User(Base):
    __tablename__ = 'users'

    user_id : Mapped[uuid.UUID] = mapped_column(UUID , primary_key=True , default=uuid.uuid4)
    username : Mapped[str] = mapped_column(String(50) , unique=True , nullable=False)
    hash_pass : Mapped[str] = mapped_column(String(255) , nullable = False)

    # role
    role : Mapped[UserRole] = mapped_column(
        Enum(UserRole , name = 'user_role' , create_type = True),
        default=UserRole.USER,
        nullable = False
    )
    owned_threads : Mapped[list[Thread]] = relationship(back_populates='creator' , lazy = 'noload')

