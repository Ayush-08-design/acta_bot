import logging
import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ...models.orm.user import User, UserRole

logger = logging.getLogger(__name__)


class UserAlreadyExistsError(Exception):
    """Raised when a username or user_id already exists in the database."""


async def create_user(
    db: AsyncSession,
    username: str,
    hash_pass: str,
    user_id: uuid.UUID,
) -> User:
    """
    Creates and persists a new user with a normalized username.
    Raises UserAlreadyExistsError on duplicate username or user_id.
    """
    username = username.strip().lower()
    if not username:
        raise ValueError("Username cannot be empty.")

    obj = User(
        user_id=user_id or uuid.uuid4(),
        username=username,
        hash_pass=hash_pass,
        role=UserRole.USER,
    )
    try:
        db.add(obj)
        await db.commit()
        await db.refresh(obj)
        return obj
    except IntegrityError as e:
        await db.rollback()
        logger.warning("Duplicate user registration attempt for '%s': %s", username, e)
        raise UserAlreadyExistsError(f"Username '{username}' is already taken.") from e
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("Database error creating user '%s'", username)
        raise


async def get_user_by_id(
    db: AsyncSession,
    user_id: uuid.UUID | str,
) -> User | None:
    """Fetch a user by primary key [user_id]"""

    user_id = uuid.UUID(user_id) if isinstance(user_id, str) else user_id

    stmt = select(User).where(User.user_id == user_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_username(db: AsyncSession, current_username: str, new_username: str):
    result = await get_user_by_username(db, new_username)
    if result:
        return False
    stmt = (
        update(User)
        .where(User.username == current_username)
        .values(username=new_username)
    )
    try:
        result = await db.execute(stmt)
        await db.commit()
        return (result.rowcount or 0) > 0
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("Failed to update username for user %s", current_username)
        raise


async def get_user_by_username(
    db: AsyncSession,
    username: str,
) -> User | None:
    """Case insensitive lookup by username (for login)."""
    username = username.strip().lower()
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_user_password(
    db: AsyncSession,
    user_id: uuid.UUID | str,
    new_hash_pass: str,
) -> bool:
    """Updates a user's hashed password. Returns True if user existed and was updated."""
    user_id = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
    stmt = update(User).where(User.user_id == user_id).values(hash_pass=new_hash_pass)
    try:
        result = await db.execute(stmt)
        await db.commit()
        return (result.rowcount or 0) > 0
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("Failed to update password for user %s", user_id)
        raise


async def update_user_role(
    db: AsyncSession,
    user_id: uuid.UUID | str,
) -> User | None:
    """Promotes a user's role (e.g., USER -> ADMIN). [For now only USER TO ADMIN]"""

    user_id = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
    new_role = UserRole.ADMIN
    stmt = (
        update(User)
        .where(User.user_id == user_id)
        .values(role=new_role)
        .returning(User)
    )
    try:
        result = await db.execute(stmt)
        await db.commit()
        return result.scalar_one_or_none()
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("Failed to update role for user %s", user_id)
        raise


async def delete_user(
    db: AsyncSession,
    user_id: uuid.UUID | str,
) -> bool:
    """Deletes a user account. Returns True if deleted, False if not found."""
    user_id = uuid.UUID(user_id) if isinstance(user_id, str) else user_id

    stmt = delete(User).where(User.user_id == user_id)
    try:
        result = await db.execute(stmt)
        await db.commit()
        return (result.rowcount or 0) > 0
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("Failed to delete user %s", user_id)
        raise