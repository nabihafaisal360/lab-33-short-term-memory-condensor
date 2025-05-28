from typing import Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from selective_deletition.configuration import llm_with_tools
from selective_deletition.tools import tools
from selective_deletition.state import AgentState

import json

# tool lookup
tools_by_name = {tool.name: tool for tool in tools}

# Tool node
def tool_node(state: AgentState) -> dict:
    print("--- Node: Tools Execution ---")
    outputs = []
    # Ensure last message is an AIMessage and has tool_calls
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        print("Tool node: Last message is not an AIMessage with tool calls. Skipping.")
        return {} 

    for tool_call in last_message.tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    # The custom reducer will handle appending these to the main messages list
    return {"messages": outputs}

# llm_with_tools node
def call_llm_with_tools(state: AgentState, config: RunnableConfig) -> dict:
    print(f"--- Node: Agent (LLM Call) ---")
    system_prompt = SystemMessage(
        content="You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    )
    # The custom_messages_reducer in AgentState handles the actual list of messages
    # This node just provides new messages to be appended by the reducer.
    response = llm_with_tools.invoke([system_prompt] + state["messages"], config)
    print(f"LLM Response: {response.content[:80]}...")
    # The custom reducer will handle appending this to the main messages list
    return {"messages": [response]}

# New node for deleting messages
def delete_messages_node(state: AgentState) -> dict | None:
    print("--- Node: Delete Messages Check ---")
    messages = state["messages"]
    # Only proceed if there are messages to avoid errors on empty list
    if messages and len(messages) > 2:
        print(f"Message count ({len(messages)}) > 2. Removing earliest two messages.")
        # The custom reducer will process these RemoveMessage instructions
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    print("Message count <= 2. No messages removed.")
    return None # Or return {} if all nodes must return a dict

# Conditional logic
def should_continue(state: AgentState) -> Literal["tools", "delete_messages_step"]:
    print(f"--- Condition: Should Continue? ---")
    messages = state["messages"]
    if not messages:
        # This case should ideally not be hit if inputs are handled correctly.
        # If it is, ending might be safest, but here we route to delete_messages for consistency.
        print("No messages in state. Routing to delete_messages_step (will do nothing). ")
        return "delete_messages_step"
    
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        print("Decision: AI Message with tool calls. Routing to tools.")
        return "tools"
    
    # If no tool calls, proceed to message deletion check
    print("Decision: No tool calls or not AIMessage. Routing to delete_messages_step.")
    return "delete_messages_step"

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_llm_with_tools)
workflow.add_node("tools", tool_node)
workflow.add_node("delete_messages_step", delete_messages_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "delete_messages_step": "delete_messages_step",
    },
)

# After tools, always go to delete_messages_step
workflow.add_edge("tools", "agent")

# After delete_messages_step, decide if we should loop or end.
# For this example, we'll always end after deletion. 
# A more complex loop might re-evaluate or go back to agent based on some condition.
workflow.add_edge("delete_messages_step", END)

graph = workflow.compile()
print("Graph compiled with delete_messages_node.")