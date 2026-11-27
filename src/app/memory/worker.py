import asyncio
import logging

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel

from .long_term.episodic import store_episodic_memory
from .long_term.semantic import store_semantic_memory

logger = logging.getLogger("memory_worker")


class StandardMessage(BaseModel):
    query: str
    response: str


# main function for extracting both type of memories


async def extract_and_save_memories(
    messages: list[StandardMessage] | str, store: BaseStore, config: RunnableConfig
):
    """
    Background worker that parses the messages extract info and pushes updates
    to both semantic and episodic namespaces.
    [Input]
        messages: list[StandardMessage] or a str, StrandardMessage contains user query and given response
    """
    user_id = config["configurable"].get("user_id")
    thread_id = config["configurable"].get("thread_id")

    if not user_id or not messages:
        logger.warning(
            "Skipping memory extraction: Missing user_id or conversation context."
        )
        return
    messages = (
        messages
        if isinstance(messages, str)
        else "/n".join(
            [f"User: {msg.query}, Assistant: {msg.response}" for msg in messages]
        )
    )

    try:
        logger.info(f"Triggering background memory updates for user {user_id}...")

        # Execute both extraction tasks concurrently using asyncio.gather
        stored_episodic, updated_semantic = await asyncio.gather(
            store_episodic_memory(
                user_id=user_id,
                thread_id=thread_id,
                messages=messages,
                store=store,
            ),
            store_semantic_memory(
                user_id=user_id,
                thread_id=thread_id,
                messages=messages,
                store=store,
            ),
        )

        logger.info(
            f"Memory Job Complete: Logged {stored_episodic} episodic events. "
            f"Refreshed semantic profile categories: {updated_semantic}."
        )

    except asyncio.CancelledError:
        logger.error("Asyncio error")

    except Exception:
        logger.exception("Background memory extraction failed for user_id=%s", user_id)
