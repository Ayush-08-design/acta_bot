import uuid
from datetime import UTC, datetime

from langchain_ollama import ChatOllama
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field


class EpisodicEvent(BaseModel):
    """Represents a single episodic event, milestone, or log-worthy action from a conversation turn."""

    content: str = Field(
        description="A concise summary of what happened, written in the past tense (e.g., 'The user encountered a Docker build timeout error on step 4.', 'The user successfully deployed the backend service to AWS.')."
    )
    importance_score: int = Field(
        description="A rating from 1 to 5 indicating how critical this event is for future context (1 = trivial detail, 5 = major project milestone/critical bug)."
    )


class EpisodicMemoryExtractor(BaseModel):

    events: list[EpisodicEvent] = Field(
        description="A list of chronological events extracted from the conversation log."
    )

async def store_episodic_memory(
    user_id: str,
    thread_id: str,
    messages: str,
    store: BaseStore,
) -> int:
    """
    Extracts core chronological events from a dialog stream and saves them
    using random UUID keys into the user's episodic memory store namespace.

    Returns:
        int: The number of new episodic memories stored.
    """

    llm = ChatOllama(model="qwen2.5:0.5")

    system_prompt = (
        "You are an advanced logging system for an AI Agent. "
        "Your task is to analyze the recent conversation transcript and extract specific episodic memories. "
        "An episodic memory is a significant action, event, tool output, or obstacle encountered during the session. "
        "Ignore idle greeting chatter, syntax syntax fixes, or background facts (save background facts for semantic profiling). "
        "Focus purely on milestones, actions taken, project changes, errors triggered, or solutions found."
    )

    llm = llm.with_structured_output(EpisodicMemoryExtractor)

    extraction_plan: EpisodicMemoryExtractor = await llm.ainvoke(
        [
            ("system", system_prompt),
            ("user", f"Recent Conversation Transcript:\n{messages}"),
        ]
    )

    namespace = ("users", user_id, "episodic")

    stored_count = 0
    current_time_str = datetime.now(UTC).isoformat()

    async with store:
        for event in extraction_plan.events:
            memory_id = str(uuid.uuid4())

            memory_value = {
                "content": event.content, 
                "importance": event.importance_score,
                "source_thread_id": thread_id, 
                "timestamp": current_time_str,
                "category": "episodic_log",
            }
            await store.aput(namespace=namespace, key=memory_id, value=memory_value)
            stored_count += 1

    return stored_count


async def fetch_episodic_memory(
    user_id: str, query: str, store: BaseStore
):
    async with store:
        episodes = await store.asearch(
            namespace=("users", user_id, "episodic"),
            query=query,  
            limit=3,
        )
    results = []
    for episode in episodes:
        result = {}
        result["time"] = episode.value["timestamp"]
        result["content"] = episode.value["content"]
        result["score"] = episode.score
        results.append(result)
    return result


async def delete_episodic_memory(
    user_id: str,
    key: str,
    store: BaseStore,
):
    namespace = ("users", user_id, "episodic")
    async with store:
        try:
            await store.adelete(namespace, key)
        except Exception as e:
            return e
