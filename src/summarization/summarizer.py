from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from summarization.configuration import llm  # USE YOUR GEMINI MODEL

def summarize_messages(messages_text, model=None):
    """
    Summarize a string conversation history using Gemini (via LangChain's summarization chain).
    """
    if model is None:
        model = llm  # Use your Gemini instance
    docs = [Document(page_content=messages_text)]
    summarize_chain = load_summarize_chain(model, chain_type="stuff")
    return summarize_chain.run(docs).strip()