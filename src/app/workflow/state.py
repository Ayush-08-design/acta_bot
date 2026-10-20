from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


def add_summary(old_summary : str , new_summary : str):
    if not new_summary:
        return old_summary
    return new_summary

class MainState(TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages]
    summary : Annotated[str , add_summary]
    
