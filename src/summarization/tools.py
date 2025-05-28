from langchain_core.tools import tool

@tool
def get_weather(location: str):
    """Call to get the weather from a specific location."""
    # Simulated logic (replace with real API if desired)
    if any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    else:
        return f"I am not sure what the weather is in {location}"

tools = [get_weather]      