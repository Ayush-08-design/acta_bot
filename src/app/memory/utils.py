import logging

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)


async def get_old_state(config: RunnableConfig):
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple:
            latest_state = checkpoint_tuple.checkpoint["channel_values"]
            checkpoint_id = checkpoint_tuple.config["configurable"]["checkpoint_id"]

            return {"latest_state": latest_state, "checkpoint_id": checkpoint_id}
        else:
            return {}


async def delete_thread_from_sqlite(thread_id: str):
    try:
        async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
            await checkpointer.adelete_thread(thread_id)
    except Exception:
        logger.exception("Failed to delete thread %s", thread_id)
