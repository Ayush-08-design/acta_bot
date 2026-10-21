from langchain.tools import tool
from langgraph.types import interrupt


@tool
def ask_for_help(question : str) -> str:
    """Use this tool when you need clarification, explicit approval, or 
    additional information from a human before proceeding."""
    # interrupt() pauses execution and sends the question to the client / user
    human_response = interrupt({
        'type' : 'clarification_needed',
        'message' : question
    })
    return f'Human clarification recieved : {human_response}'
