from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

models = Literal["fast-cheap", "reasoning", "coder"]

model_mapping: dict[str, BaseChatModel] = {
    "fast-cheap": ChatOllama(model="qwen2.5:0.5b"),
    "reasoning": ChatGroq(model="openai/gpt-oss-20b", temperature=1),
    "coder": ChatGroq(model="openai/gpt-oss-120b", temperature=1),
}

def get_model(tools: list, choice: models = "fast-cheap"):
    default_model: BaseChatModel = model_mapping.get(choice)
    fallback_models = []
    for name , model in model_mapping.items():
        if name != choice:
            fallback_models.append(model.bind_tools(tools))
    with_fallback = default_model.bind_tools(tools).with_fallbacks(fallback_models)
    return with_fallback
