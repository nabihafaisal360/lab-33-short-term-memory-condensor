# src/selective_deletition/state.py
from typing import (
    Annotated,
    List, # Using List[BaseMessage] for more specific typing
    TypedDict,
    # Sequence # Not strictly needed if using List consistently
)
from langchain_core.messages import BaseMessage # RemoveMessage is handled by add_messages
from langgraph.graph.message import add_messages # Import add_messages

# Removed custom_message_reducer_with_delete_logic function

# --- AgentState Definition ---
class AgentState(TypedDict):
    """The state of the agent."""
    # Using add_messages, which can handle RemoveMessage objects
    messages: Annotated[List[BaseMessage], add_messages]
