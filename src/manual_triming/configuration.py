import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from .tools import tools

# Load environment variables from a .env file if it exists
# Construct the path to the .env file, assuming it's in the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- General Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # Making this a warning for now, as it might not be strictly needed for all flows
    # or could be configured directly in init_chat_model depending on specific LangChain versions/setups
    print(f"Warning: GOOGLE_API_KEY environment variable not set. Searched for .env at {dotenv_path}")

# --- LLM Configuration ---
# The user previously had "google_genai:gemini-2.0-flash" directly in init_chat_model
# For consistency, let's use MODEL_NAME, but we'll ensure the current model is used.
MODEL_NAME = "google_genai:gemini-2.0-flash" # Matching the existing init_chat_model call
TEMPERATURE = 0.0 # Default from previous versions

# --- Memory Configuration ---
MAX_MESSAGES =4 # The maximum number of messages to keep in history

# --- Tool Configuration ---
# (Add any specific tool configs here if needed)

# Existing LLM and tools initialization
llm = init_chat_model(MODEL_NAME, client_options={"api_key": GOOGLE_API_KEY} if GOOGLE_API_KEY else {})
llm_with_tools = llm.bind_tools(tools)
