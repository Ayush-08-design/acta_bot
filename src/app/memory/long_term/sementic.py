from typing import Literal

from langchain_core.messages import BaseMessage
from langchain_core.stores import BaseStore
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

SemanticKeyType = Literal["user_profile", "tech_stack", "development_preferences"]

class SemanticKeyUpdate(BaseModel):
    """Represents a clean merge instruction for a specific custom storage key."""

    key: SemanticKeyType = Field(
        description="The target storage key folder where this information must be saved."
    )
    merged_content: str = Field(
        description="The newly synthesized block of facts for this key. You MUST merge the new conversation details into the existing state provided. Do not drop or delete old details unless the user explicitly corrected or changed them."
    )

class SemanticMemoryPlan(BaseModel):

    updates: list[SemanticKeyUpdate] = Field(
        description="A list of updates for keys that need a refresh based on the conversation."
    )

async def store_semantic_memory(
    user_id: str,
    thread_id: str,
    recent_dialogue: list[BaseMessage],
    store: BaseStore,
) -> list[str]:
    """
    Reads existing user profile states, passes them to the LLM along with the
    live dialogue, and saves the cleanly merged updates back to fixed custom keys.

    Returns:
        List[str]: A list of keys that were updated during this operation.
    """
    llm = ChatOllama(model="qwen2.5:0.5")

    namespace = ("users", user_id, "semantic")
    allowed_keys = ["user_profile", "tech_stack", "development_preferences"]

    current_memory = {}
    for k in allowed_keys:
        record = await store.aget(namespace, k)
        if record and "content" in record.value:
            current_memory[k] = record.value["content"]
        else:
            current_memory[k] = "No data recorded yet."

    formatted_dialogue = "\n".join(
        [f"{msg.type.capitalize()}: {msg.content}" for msg in recent_dialogue]
    )

    system_prompt = f"""
        You are a memory consolidation engine. Your task is to update a user's long-term semantic profile.
        You will be provided with:
        1. The CURRENT snapshot of what we already know about the user across specific keys.
        2. The RECENT conversation transcript.

        CURRENT PROFILE SNAPSHOT:
        - user_profile: {current_memory['user_profile']}
        - tech_stack: {current_memory['tech_stack']}
        - development_preferences: {current_memory['development_preferences']}

        CRITICAL INSTRUCTIONS:
        - Review the transcript for new, long-term foundational facts about the user's identity, tech choices, or coding styles.
        - If a category needs an update, generate a 'merged_content' string that appends the new information directly into the existing snapshot string.
        - NEVER delete or drop facts present in the snapshot unless the user explicitly corrects or contradicts them.
        - If a category was not mentioned or changed in the transcript, DO NOT include an update for that key.
        """

    llm = llm.with_structured_output(SemanticMemoryPlan)

    plan: SemanticMemoryPlan = await llm.ainvoke(
        [
            ("system", system_prompt),
            ("user", f"Recent Conversation Transcript:\n{formatted_dialogue}"),
        ]
    )

    async with store:
        updated_keys = []
        for update in plan.updates:
            memory_value = {
                "content": update.merged_content, 
                "source_thread_id": thread_id,
                "category": "semantic_profile",
            }

            await store.aput(namespace=namespace, key=update.key, value=memory_value)
            updated_keys.append(update.key)

    return updated_keys


async def fetch_sementic_memory(
    user_id: str,
    key: SemanticKeyType,
    store: BaseStore,
):
    async with store:
        namespace = ("users", user_id, "sementic")
        record = await store.aget(namespace, key)
        result = record.value["content"]
        return result


async def delete_sementic_memory(
    user_id: str,
    key: SemanticKeyType,
    store: BaseStore,
):
    async with store:
        namespace = ("users", user_id, "sementic")
        try:
            await store.adelete(namespace, key)
        except Exception as e:
            return e
