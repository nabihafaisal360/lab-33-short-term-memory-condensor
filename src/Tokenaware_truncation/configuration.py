from langchain.chat_models import init_chat_model
from Tokenaware_truncation.tools import tools

llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

# Maximum tokens to keep in the message history before truncation
MAX_TOKENS_FOR_HISTORY = 500