from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
#from langgraph.checkpoint.memory import MemorySaver

from summarization.configuration import llm_with_tools, llm
from summarization.tools import tools
from summarization.utils import messages_to_str
from summarization.summarizer import summarize_messages
from summarization.state import AgentState # Ensure AgentState is imported from state.py

import json

# === Parameters for summarization logic ===
MAX_MESSAGES_BEFORE_SUMMARY = 4      # When to summarize
NUM_RECENT_FOR_CONTEXT = 2           # Recent messages to keep in detail
SUMMARY_MSG_PREFIX = "Summary of previous conversation: " # For identifying summary messages

# === State Definition (inc. summary) ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], ...]  # reduce with add_messages

# Set up the message reducer for LangGraph
from langgraph.graph.message import add_messages
AgentState.__annotations__["messages"] = Annotated[Sequence[BaseMessage], add_messages]

# === Tool lookup helper ===
tools_by_name = {tool.name: tool for tool in tools}

# === Node: Summarize Conversation (Renamed from summarize_messages_node) ===
def summarize_conversation_node(state: AgentState) -> dict:
    print("--- Node: Summarize Conversation ---")
    messages = state["messages"]
    messages_to_summarize = messages[:-NUM_RECENT_FOR_CONTEXT]
    recent_messages = messages[-NUM_RECENT_FOR_CONTEXT:]
    history_text = messages_to_str(messages_to_summarize)
    print(f"Summarizing {len(messages_to_summarize)} messages. Keeping {len(recent_messages)} recent messages.")
    new_summary_text = summarize_messages(history_text, model=llm)
    summary_message = SystemMessage(content=f"{SUMMARY_MSG_PREFIX}{new_summary_text}")
    print(f"New summary created: {summary_message.content[:100]}...")
    updated_messages = [summary_message] + recent_messages
    return {"messages": updated_messages}

# === Node: Conversation (Main LLM Agent Call - Renamed from agent_node) ===
def conversation_node(state: AgentState, config: RunnableConfig) -> dict:
    print("--- Node: Conversation (LLM Call) ---")
    messages_for_llm = [
        SystemMessage(content="You are a helpful AI assistant, please respond to the user's query to the best of your ability!")
    ] + state["messages"]
    print(f"Calling LLM with {len(messages_for_llm)} messages. First state message: {state['messages'][0].content[:60] if state['messages'] else '[No messages yet]'}...")
    response = llm_with_tools.invoke(messages_for_llm, config)
    print(f"LLM Response: {response.content[:80]}...")
    return {"messages": [response]}

# === Node: Tools Execution (Renamed from tool_node) ===
def tools_node(state: AgentState) -> dict:
    print("--- Node: Tools Execution ---")
    outputs = []
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        print("Tools Node: No tool calls in last AI message. Should not happen if routed here.")
        return {}
    print(f"Tools Node: Processing {len(last_message.tool_calls)} tool call(s).")
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_call_id = tool_call.get("id")
        if not all([tool_name, tool_call_id]):
            print(f"Tools Node: Invalid tool call structure: {tool_call}, skipping.")
            continue
        print(f"Tools Node: Invoking tool '{tool_name}' with args: {tool_args}")
        try:
            tool_result = tools_by_name[tool_name].invoke(tool_args)
            outputs.append(ToolMessage(content=json.dumps(tool_result), name=tool_name, tool_call_id=tool_call_id))
            print(f"Tools Node: Tool '{tool_name}' success.")
        except Exception as e:
            print(f"Tools Node: Tool '{tool_name}' error: {e}")
            outputs.append(ToolMessage(content=json.dumps({"error": str(e)}), name=tool_name, tool_call_id=tool_call_id))
    return {"messages": outputs}

# === Conditional Edge: Route after Conversation Node ===
def route_from_conversation_node(state: AgentState) -> Literal["tools", "summarize_conversation", END]:
    print("--- Condition: Route from Conversation Node ---")
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        print("Decision: Agent requested tool calls. Routing to Tools node.")
        return "tools"
    
    # If no tools, check for summarization based on message count
    if len(state["messages"]) > MAX_MESSAGES_BEFORE_SUMMARY:
        print(f"Decision: No tools. Message count ({len(state['messages'])}) > MAX_MESSAGES ({MAX_MESSAGES_BEFORE_SUMMARY}). Routing to Summarize.")
        return "summarize_conversation"
    
    print(f"Decision: No tools. Message count ({len(state['messages'])}) <= MAX_MESSAGES ({MAX_MESSAGES_BEFORE_SUMMARY}). Routing to END.")
    return END

# === Build the LangGraph workflow ===
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("conversation", conversation_node)
workflow.add_node("tools", tools_node)
workflow.add_node("summarize_conversation", summarize_conversation_node)

# Set entry point
workflow.set_entry_point("conversation")

# Define edges
workflow.add_conditional_edges(
    "conversation",
    route_from_conversation_node,
    {
        "tools": "tools",
        "summarize_conversation": "summarize_conversation", # Route directly to summarize_conversation
        END: END  # Route directly to END
    }
)

workflow.add_edge("tools", "conversation") # Loop back to conversation after tools

workflow.add_edge("summarize_conversation", END)

# Add checkpointer
#memory = MemorySaver()
graph = workflow.compile()

print("Graph compiled with simplified structure and checkpointer.")

# Example of how to run (for user reference, not executed by assistant):
# async def main_run():
#     config = {"configurable": {"thread_id": "user_123"}} # thread_id is important for checkpointer
#     inputs = [ 
#         {"messages": [HumanMessage(content="Hi there!")]},
#         {"messages": [HumanMessage(content="What is the weather like in San Francisco?")]}, 
#         {"messages": [HumanMessage(content="Thanks!")]},
#         {"messages": [HumanMessage(content="Tell me another fact about weather.")]}, 
#         {"messages": [HumanMessage(content="Okay, one more: how about New York weather?")]},
#         {"messages": [HumanMessage(content="Great, that is all for now.")]},
#     ]
#     current_input_messages = []
#     for i, turn_input in enumerate(inputs):
#         print(f"\n--- RUNNING TURN {i+1} --- INPUT: {turn_input['messages'][0].content}")
#         # For a streaming effect with checkpointer, you usually invoke with new messages.
#         # The checkpointer handles loading previous state for the given thread_id.
#         # The input to invoke should be the new message(s) for this turn.
#         # However, our AgentState is defined as `messages: Annotated[Sequence[BaseMessage], add_messages]`
#         # So, the graph expects the full current list or just new additions which add_messages handles.
#         # For simplicity in this example, let's pass only the new message for this turn as input, 
#         # assuming `add_messages` handles accumulation based on checkpointed state.
#         result = graph.invoke(turn_input, config=config)
#         print(f"Turn {i+1} complete. Final state messages: {len(result['messages'])}")
#         for msg_idx, msg in enumerate(result["messages"]):
#             print(f"  Msg {msg_idx} ID: {msg.id} ({type(msg).__name__}) {msg.content[:70]}...")
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main_run())