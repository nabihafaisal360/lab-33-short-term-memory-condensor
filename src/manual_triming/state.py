from typing import (
    Annotated,
    Sequence,
    TypedDict,
    Union,
    List
)
from langchain_core.messages import BaseMessage
# Removed: from langgraph.graph.message import add_messages
# We will define our own reducer.

from manual_triming.configuration import MAX_MESSAGES


def manage_messages_history(
    existing: Sequence[BaseMessage],
    updates: Union[Sequence[BaseMessage], dict],
) -> Sequence[BaseMessage]:
    """
    Manages the message history, adding new messages and trimming old ones.
    """
    if isinstance(updates, dict):
        if updates.get("type") == "trim":
            # Trim messages, keeping only the last MAX_MESSAGES
            # If from_end is specified, it means how many messages to keep from the end.
            # This is slightly different from your example's "from" and "to"
            # but aligns with keeping the "last N".
            num_to_keep = updates.get("count", MAX_MESSAGES)
            return existing[-num_to_keep:]
        # Potentially handle other dictionary-based update types here
        # For now, if it's a dict not for trimming, we raise an error or return existing.
        # Or, if other dict updates are expected, add logic for them.
        # For safety, let's assume only 'trim' is a valid dict update for now.
        raise ValueError(f"Unsupported dictionary update type for messages: {updates}")
    elif isinstance(updates, list): # Langchain typically appends lists of BaseMessage
        # Add new messages
        combined = list(existing) + list(updates)
        # Trim if history exceeds MAX_MESSAGES
        if len(combined) > MAX_MESSAGES:
            return combined[-MAX_MESSAGES:]
        return combined
    else:
        # This case should ideally not be reached if types are correct
        raise TypeError(
            f"Unsupported update type for messages: {type(updates)}. Expected Sequence[BaseMessage] or dict."
        )


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], manage_messages_history]