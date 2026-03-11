import re
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from utils.llm import get_llm

SYSTEM_PROMPT = (
    "You are a helpful search assistant. Use the 'browser_search' tool to find current information "
    "on the web to answer the user's query accurately. "
    "Be concise, answer in short paragraphs, and use bullets only for lists. "
    "Always cite your sources at the end: Sources:\n- [Title](URL)"
)


def clean_chunk(text: str) -> str:
    """Cleans up any remaining citation markers and HTML tags from a chunk."""
    text = re.sub(r"【.*?】", "", text)
    text = text.replace("<br>", "\n")
    return text


def search_stream(query: str, chat_history: list = None):
    """Streams the search response by executing a tool call if needed."""
    llm = get_llm()
    
    # Initialize search tool
    search = DuckDuckGoSearchRun()
    
    # Define tool for LLM in a more standard way
    tools = [
        {
            "name": "browser_search",
            "description": "Search the web for real-time information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the internet."
                    }
                },
                "required": ["query"]
            }
        }
    ]
    
    # Using bind_tools with tool_choice="auto" is often more stable for Groq
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if chat_history:
        for msg in chat_history[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    
    messages.append(HumanMessage(content=query))
    
    try:
        # Step 1: Get tool call from LLM
        response = llm_with_tools.invoke(messages)
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            search_query = tool_call["args"].get("query", query)
            
            # Step 2: Execute search
            with st.status(f"Searching for: {search_query}...", expanded=False) as status:
                search_results = search.run(search_query)
                status.update(label="Search complete!", state="complete")
            
            # Step 3: Feed results back to LLM
            messages.append(response)
            messages.append(AIMessage(content=f"Search Results:\n{search_results}", tool_call_id=tool_call["id"]))
            
            # Step 4: Stream final answer
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield clean_chunk(chunk.content)
        else:
            # Fallback: model didn't use tool, just stream content if any
            if response.content:
                yield clean_chunk(response.content)
            else:
                # If everything fails, just call llm directly
                for chunk in llm.stream(messages):
                    if chunk.content:
                        yield clean_chunk(chunk.content)
                        
    except Exception as e:
        yield f"Error in search: {str(e)}"
