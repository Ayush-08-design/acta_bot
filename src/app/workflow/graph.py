from langchain.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.messages.utils import trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.stores import BaseStore
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src.app.llm.guardrails.utils import guarded_tool_wrapper
from src.app.llm.model import get_model
from src.app.workflow.instruction import get_instructions
from src.app.workflow.state import MainState, SummarySchema

instructions = get_instructions()


async def build_workflow(
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    tools: list[BaseTool],
    system_message: str = instructions,
):
    tool_dict = {tool.name: tool for tool in tools}

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
        summary = state["summary"] if hasattr(state, "summary") else None

        existing_summary = ""
        if summary:
            existing_summary = summary

        retained_messages = msg_trimmer.invoke(all_messages)

        retained_ids = {msg.id for msg in retained_messages if msg.id}

        trimmed_messages = [msg for msg in all_messages if msg.id not in retained_ids]

        if not trimmed_messages:
            return {"summary": ""}

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
            response["summmary"] if isinstance(response, dict) else response.summary
        )

        state_updates = []

        for msg in trimmed_messages:
            if msg.id:
                state_updates.append(RemoveMessage(id=msg.id))

        return {"messages": state_updates, "summary": summary}

    async def reasoner(state: MainState, config: RunnableConfig):

        cfg = config.get("configurable", {})

        # Fetch model choice and other metadata from configurable build from data send by the user
        model_choice = cfg.get("model_choice", "fast-cheap")

        tools_enabled = cfg.get("tools_enabled", True)
        allowed_tools = cfg.get("allowed_tools")

        if tools_enabled:
            active_tools = (
                [tool_dict[name] for name in allowed_tools if name in tool_dict]
                if allowed_tools is not None
                else tools
            )

        model = get_model(active_tools, model_choice)
        nonlocal system_message

        messages = state["messages"]
        summary = state["summary"] if hasattr(state, "summary") else None
        if summary:
            system_message += f"\n\n[SUMMARY OF OLDER CONVERSATION]:\n{summary}"
        system_message = SystemMessage(content=system_message)
        messages = [system_message] + messages

        try:
            response = await model.ainvoke(messages)
        except Exception as e:  
            response = AIMessage(
                content=f"Internal error occured please try again after some time : {e}"
            )
        return {"messages": [response]}

    # Check if the model decided it needs to use the tool
    async def tool_condition(state: MainState):
        msg = state["messages"][-1]
        if msg.tool_calls:
            return "tool call"
        return "end"

    async def tool_node(state: MainState, config: RunnableConfig, *, store: BaseStore):
        tool_outputs = []
        ai_msg = state["messages"][-1]
        for tool_call in ai_msg.tool_calls:
            if "store" in tool_call["args"]:
                tool_call["args"]["store"] = store
            if "config" in tool_call["args"]:
                tool_call["args"]["config"] = config
                
            tool_output = await tool_dict[tool_call["name"]].ainvoke(tool_call)
            safe_tool_output = guarded_tool_wrapper(tool_output)
            tool_outputs.append(safe_tool_output)
            
        return {"messages": [tool_output for tool_output in tool_outputs]}

    graph = StateGraph(MainState)

    graph.add_node("context_manager", context_manager)
    graph.add_node("reasoner", reasoner)
    graph.add_node("tool_node", tool_node)

    graph.add_edge(START, "context_manager")
    graph.add_edge("context_manager", "reasoner")
    graph.add_conditional_edges(
        "reasoner", tool_condition, path_map={"tool call": "tool_node", "end": END}
    )
    graph.add_edge("tool_node", "context_manager")
    workflow = graph.compile(checkpointer=checkpointer, store=store)

    return workflow
