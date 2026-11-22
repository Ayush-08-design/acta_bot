import ast

from langchain.tools import tool


@tool
async def calculator(query: str) -> str:
    "A simple calculator , input should be a valid mathmetical expression"
    return ast.literal_eval(query)
