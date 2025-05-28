from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage, trim_messages
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from Tokenaware_truncation.configuration import llm_with_tools, llm, MAX_TOKENS_FOR_HISTORY
from Tokenaware_truncation.tools import tools
from Tokenaware_truncation.state import AgentState

import json

# tool lookup
tools_by_name = {tool.name: tool for tool in tools}

# Tool node
def tool_node(state: AgentState) -> dict:
    outputs = []
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
    return {"messages": outputs}

# llm_with_tools node
def call_llm_with_tools(state: AgentState, config: RunnableConfig) -> dict:
    print(f"--- Node: Agent (LLM Call) ---")
    current_messages = state["messages"]
    
    system_prompt_content = "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    
    processed_messages = []
    if not current_messages or not (isinstance(current_messages[0], SystemMessage) and current_messages[0].content == system_prompt_content):
        processed_messages.append(SystemMessage(content=system_prompt_content))
    
    processed_messages.extend(current_messages)

    print(f"Original message count for trimming: {len(processed_messages)}")

    trimmed_messages = trim_messages(
        processed_messages,
        max_tokens=MAX_TOKENS_FOR_HISTORY,
        strategy="last",
        token_counter=llm,
        include_system=True,
        start_on="human",
        end_on=("human", "tool"),
    )
    print(f"Trimmed message count: {len(trimmed_messages)}")
    print(f"Content of trimmed_messages: {trimmed_messages}")

    response = llm_with_tools.invoke(trimmed_messages, config)
    print(f"LLM Response: {response.content[:80]}...")
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", END]:
    print(f"--- Condition: Should Continue? ---")
    messages = state["messages"]
    if not messages:
        print("No messages in state. Ending.")
        return END
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        print("Decision: AI Message with tool calls. Routing to tools.")
        return "tools"
    print("Decision: No tool calls or not AI message. Ending.")
    return END

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_llm_with_tools)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)
workflow.add_edge("tools", "agent")
graph = workflow.compile()
print("Graph compiled with token-aware truncation.")