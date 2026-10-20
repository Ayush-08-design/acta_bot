from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.messages.utils import trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.stores import BaseStore
from langchain_core.tools.structured import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src.app.llm.model import get_model
from src.app.llm.prompt_templates import default_chat_prompt
from src.app.models.schemas.common import SummarySchema
from src.app.tools import tools
from src.app.workflow.state import MainState


async def get_workflow(
    config: RunnableConfig,
    checkpointer : BaseCheckpointSaver,
    store : BaseStore,
    model_choice: str | None = "fast-cheap",
    system_message: str = default_chat_prompt,
):
    model = get_model(tools, model_choice)

    async def context_manager(state: MainState):

        msg_trimmer = trim_messages(
            max_tokens=16000,
            strategy="last",
            token_counter="approximate",
            start_on="human",
            end_on="human",
            include_system=False,  
        )
    
        all_messages = state["messages"]
        summary = state['summary'] if hasattr(state , 'summary') else None

        existing_summary = "" 
        if summary:
            existing_summary = summary
            
        retained_messages = msg_trimmer.invoke(all_messages)

        retained_ids = {msg.id for msg in retained_messages if msg.id}

        trimmed_messages = [msg for msg in all_messages if msg.id not in retained_ids]

        if not trimmed_messages:
            return {"summary": ''}

        formatted_trimmed_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in trimmed_messages
        )
        summary_prompt = f"""
            You are an internal system agent. Consolidate the following context into a concise bulleted summary.
            Prior Summary Context: {existing_summary}
            New updates to roll in:
            {formatted_trimmed_text}
        """
        model = ChatOllama(model="qwen2.5:0.5b")
        
        response = await model.with_structured_output(SummarySchema).ainvoke(
            [HumanMessage(content=summary_prompt)]
        )
        summary = (
            response["summmary"] if isinstance(response, dict) else response.summmary
        )

        state_updates = []

        for msg in trimmed_messages:
            if msg.id:
                state_updates.append(RemoveMessage(id=msg.id))

        return {"messages": state_updates , 'summary' : summary}

    async def reasoner(state: MainState):
        nonlocal system_message

        messages = state["messages"]
        summary = state['summary'] if hasattr(state , 'summary') else None
        if summary:
            system_message += f"\n\n[SUMMARY OF OLDER CONVERSATION]:\n{summary}"
        system_message = SystemMessage(content=system_message)
        messages = [system_message] + messages

        try:
            response = await model.ainvoke(messages)
        except Exception as e:  # noqa: BLE001
            response = AIMessage(
                content=f"Internal error occured please try again after some time : {e}"
            )
        return {"messages": [response]}

    # 5. Check if the model decided it needs to use the tool
    async def tool_condition(state: MainState):
        msg = state["messages"][-1]
        if msg.tool_calls:
            return "tool call"
        return "end"

    async def create_tool_dict(
        tools: list[StructuredTool],
    ) -> dict[str, StructuredTool]:
        tool_dict = {}
        for tool in tools:
            tool_dict[tool.get_name()] = tool
        return tool_dict

    async def tool_node(state: MainState , config : RunnableConfig , * , store: BaseStore):
        tool_dict = await create_tool_dict(tools)
        tool_outputs = []
        ai_msg = state["messages"][-1]
        for tool_call in ai_msg.tool_calls:
            if "store" in tool_call["args"]:
                tool_call["args"]["store"] = store
            if 'config' in tool_call["args"]:
                tool_call["args"]["config"] = config
                
            tool_outputs.append(await tool_dict[tool_call["name"]].ainvoke(tool_call))
        return {"messages": [tool_output for tool_output in tool_outputs]}

    graph = StateGraph(MainState)

    graph.add_node('context_manager', context_manager)
    graph.add_node("reasoner", reasoner)
    graph.add_node("tool_node", tool_node)

    graph.add_edge(START , 'context_manager')
    graph.add_edge('context_manager', "reasoner")
    graph.add_conditional_edges(
        "reasoner", tool_condition, path_map={"tool call": "tool_node", "end": END}
    )
    graph.add_edge("tool_node", 'context_manager')
    workflow = graph.compile(checkpointer = checkpointer , store = store)

    return workflow
