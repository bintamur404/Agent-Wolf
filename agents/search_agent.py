import re
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun
from utils.llm import get_llm

from pydantic import BaseModel, Field

SYSTEM_PROMPT = (
    "You are Agent Wolf, a tactical search specialist. Your primary directive is to use the 'browser_search' tool for any query that requires external information. "
    "CRITICAL: If you need to use the 'browser_search' tool, output ONLY the tool call. Do not provide any conversational preamble, explanation, or 'memory-based' answer. "
    "Only after you receive the search results should you synthesize a final response. "
    "Final responses must be concise, use source citations, and follow this format: Sources:\n- [Title](URL)"
)

class browser_search(BaseModel):
    """Search the web for real-time information."""
    query: str = Field(description="The hunt-query to search the web for.")


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
    
    # Bind tools using Pydantic (most robust for Groq)
    llm_with_tools = llm.bind_tools([browser_search])
    
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
        
        # Groq sometimes returns tool_calls in different places, check both
        tool_calls = getattr(response, 'tool_calls', [])
        search_query = None
        tool_call_id = None
        
        if tool_calls:
            tool_call = tool_calls[0]
            # Use the args from the Pydantic model
            search_query = tool_call["args"].get("query", query)
            tool_call_id = tool_call["id"]
        elif response.content:
            # Fallback: model didn't use tool, just stream content if any
            yield clean_chunk(response.content)
            return
            
    except Exception as e:
        error_msg = str(e)
        search_query = None
        tool_call_id = None
        
        if "tool_use_failed" in error_msg and "failed_generation" in error_msg:
            import json
            match = re.search(r'<function=browser_search(.*?)</function>', error_msg)
            if match:
                try:
                    args = json.loads(match.group(1))
                    search_query = args.get("query", query)
                    tool_call_id = "manual_fallback"
                except Exception:
                    pass
                    
        if not search_query:
            yield f"Error in search: {error_msg}"
            return

    if search_query:
        # Step 2: Execute search
        with st.status(f"Searching for: {search_query}...", expanded=False) as status:
            search_results = search.run(search_query)
            status.update(label="Search complete!", state="complete")
        
        # Step 3: Feed results back to LLM using ToolMessage (required by Groq)
        if "response" in locals():
            messages.append(response)
        else:
            messages.append(AIMessage(content="", tool_calls=[{"name": "browser_search", "args": {"query": search_query}, "id": tool_call_id}]))
            
        messages.append(ToolMessage(content=f"Search Results:\n{search_results}", tool_call_id=tool_call_id))
        
        # Step 4: Stream final answer
        try:
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield clean_chunk(chunk.content)
        except Exception as e2:
            yield f"Error in final generation: {str(e2)}"
