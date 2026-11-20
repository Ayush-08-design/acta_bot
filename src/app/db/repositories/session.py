import logging
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.memory.utils import delete_thread_from_sqlite

from ...models.orm.thread import Thread

logger = logging.getLogger(__name__)


class ThreadConflictError(Exception):
    """Raised when a thread_id already exists under a different user or violates session constraints."""


async def create_thread(
    session: AsyncSession,
    user_id: UUID | str,
    thread_id: UUID | str,
    title: str = "New Chat",
) -> Thread:
    """
    Creates a new chat thread or returns the existing one if owned by the same user.
    Raises ThreadConflictError if the thread_id belongs to another user or user_id is invalid.
    """
    # Checking if thread already exists
    existing = await get_thread_by_id(session, thread_id=thread_id, user_id=user_id)
    if existing:
        return existing

    # Attempt creation
    obj = Thread(thread_id=thread_id, title=title, owner_id=user_id)
    try:
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return obj
    except IntegrityError as e:
        await session.rollback()
        # Handle race condition where another concurrent request created it milliseconds earlier
        existing = await get_thread_by_id(session, thread_id=thread_id, user_id=user_id)
        if existing:
            return existing

        logger.warning(
            "IntegrityError creating thread %s for user %s: %s", thread_id, user_id, e
        )
        raise ThreadConflictError(
            "Thread ID already exists for another user or user_id does not exist."
        ) from e
    except SQLAlchemyError:
        await session.rollback()
        logger.exception("Database error creating thread %s", thread_id)
        raise


async def get_thread_by_id(
    session: AsyncSession,
    thread_id: UUID | str,
    user_id: UUID | str,
) -> Thread | None:
    """Fetch a single thread, strictly scoped to its owner."""
    stmt = select(Thread).where(
        Thread.thread_id == thread_id,
        Thread.owner_id == user_id,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_user_threads(
    session: AsyncSession,
    user_id: UUID | str,
    limit: int = 20,
    offset: int = 0,
) -> list[Thread]:
    """Paginated list of a user's chat threads, ordered by most recently updated."""
    stmt = (
        select(Thread)
        .where(Thread.owner_id == user_id)
        .order_by(
            Thread.updated_at.desc()
        )  # Falls back to Thread.created_at.desc() if no updated_at
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_thread_title(
    session: AsyncSession,
    thread_id: UUID | str,
    user_id: UUID | str,
    new_title: str,
) -> Thread | None:
    """
    Renames a thread. Returns the updated Thread, or None if not found / unauthorized.
    """
    clean_title = new_title.strip()[:100]  # Enforce max length
    if not clean_title:
        raise ValueError("Thread title cannot be empty.")

    stmt = (
        update(Thread)
        .where(Thread.thread_id == thread_id, Thread.owner_id == user_id)
        .values(title=clean_title)
        .returning(Thread)
    )
    try:
        result = await session.execute(stmt)
        await session.commit()
        return result.scalar_one_or_none()
    except SQLAlchemyError:
        await session.rollback()
        logger.exception("Failed to update title for thread %s", thread_id)
        raise


async def delete_thread(
    session: AsyncSession,
    thread_id: UUID | str,
    user_id: UUID | str,
) -> bool:
    """
    Deletes a thread owned by the user. Returns True if deleted, False if not found.
    """
    stmt = delete(Thread).where(
        Thread.thread_id == thread_id,
        Thread.owner_id == user_id,
    )
    try:
        result = await session.execute(stmt)
        await session.commit()
        await delete_thread_from_sqlite(str(thread_id))
        return (result.rowcount or 0) > 0
    except SQLAlchemyError:
        await session.rollback()
        logger.exception("Failed to delete thread %s", thread_id)
        raise


async def delete_all_threads(session: AsyncSession, user_id: str):
    stmt = delete(Thread).where(Thread.owner_id == user_id)
    try:
        result = await session.execute(stmt)
        await session.commit()
        return (result.rowcount or 0) > 0
    except SQLAlchemyError:
        await session.rollback()
        logger.exception("Failed to delete threads of user %s", user_id)
        raise
