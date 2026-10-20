from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

models = Literal["fast-cheap", "reasoning", "coder"]

# Routing schema
class RoutingDecision(BaseModel):
    reasoning: str = Field(
        description="Briefly explain the rationale behind choosing this specific route."
    )
    target_model: Literal["fast-cheap", "reasoning", "coder"] = Field(
        description="Select the optimal model tier based on the query complexity and intent."
    )

# Router class
class LLMQueryRouter:
    def __init__(self):
        # Using a fast, inexpensive intelligence layer for routing decisions
        self.__model = ChatOllama(model="qwen2.5:0.5b")
        self.__response_schema = RoutingDecision
        self.__system_prompt = (
            "You are an elite triage router infrastructure component. Evaluate the user's "
            "incoming query and route it to the optimal model category based on these rules:\n\n"
            "- 'fast-cheap': For simple chit-chat, scheduling, text summaries, and low-complexity tasks.\n"
            "- 'reasoning': For advanced math, deep logical problems, strategy, and multi-step tasks.\n"
            "- 'coder': For anything explicitly asking for code, programming debugging, or system architectures."
        )

    async def route_query(self, user_query: str) -> models:
        message = [
            {"role": "system", "content": self.__system_prompt},
            {"role": "user", "content": user_query},
        ]
        model = self.__model.with_structured_output(self.__response_schema)
        response = await model.ainvoke(message)
        return response["target_model"]

    def set_prompt(self, new_prompt: str):
        self.__system_prompt = new_prompt

    def set_model(self, model):
        self.__model = model
