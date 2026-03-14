import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from agents.arxiv_agent import search_arxiv, format_papers_for_llm
from utils.llm import get_llm


LIT_REVIEW_PROMPT = """You are Agent Wolf's Research Synthesis Engine — an expert academic writer.

You have been given real academic papers from ArXiv AND recent web search results on the topic.
Write a structured **Literature Review** in exactly this format:

## 1. Background & Context
(2-3 paragraphs: What is this field? Why does it matter? Historical context.)

## 2. Key Methodologies
(Summarize the main approaches researchers use, citing papers.)

## 3. Major Findings & Results
(What has been established? Key results from the papers found.)

## 4. Current Challenges & Limitations
(What problems persist? What assumptions are challenged?)

## 5. Research Gaps & Future Directions
(THIS IS THE MOST IMPORTANT SECTION. What has NOT been studied? Where should the next paper go?)

## References
(List all cited papers with ArXiv links in APA format.)

Rules:
- Cite every claim: [Author et al., Year](url)
- Be academic but clear
- Target length: 600-900 words
- Never make up papers — only cite what was provided
"""

GAP_FINDER_PROMPT = """You are Agent Wolf's Research Gap Finder — a strategic scientific advisor.

Your mission: Analyze the provided papers and identify the most promising **unexplored research areas**.

Output format:

## 🕵️ Research Gap Analysis: [Topic]

### The Current Frontier
(1 paragraph: What is the state of the art?)

### 🔴 Critical Gaps Identified
For each gap, use this template:
**Gap [N]: [Short Title]**
- **What's missing:** (Specific description of the unstudied area)
- **Why it matters:** (Real-world Impact)
- **Suggested methodology:** (How could a researcher approach this?)
- **Potential paper title:** "A [Method] Approach to [Problem] in [Domain]"

### 🌟 Top Recommendation
(Your single strongest recommendation for the next research project, with brief justification.)

Be bold, specific, and actionable. Think like a PhD advisor.
"""


def literature_review_stream(topic: str):
    """
    Combines ArXiv search + DuckDuckGo web search, then
    streams a full structured literature review.
    """
    llm = get_llm()
    ddg = DuckDuckGoSearchRun()

    # Step 1: Search ArXiv
    with st.status("🔬 Searching ArXiv for academic papers...", expanded=False) as status:
        papers = search_arxiv(topic, max_results=6)
        status.update(label=f"✅ Found {len(papers)} ArXiv papers", state="complete")

    # Step 2: Web search for latest context
    with st.status("🌐 Grounding with latest web context...", expanded=False) as status:
        web_context = ddg.run(f"{topic} research 2024 2025 findings")
        status.update(label="✅ Web context retrieved", state="complete")

    # Step 3: Display source summary
    st.markdown(f"#### 📚 Literature Review: *{topic}*")
    st.markdown(f"*Synthesized from {len(papers)} ArXiv papers + live web sources*")
    st.divider()

    # Step 4: Stream the review
    paper_context = format_papers_for_llm(papers)
    messages = [
        SystemMessage(content=LIT_REVIEW_PROMPT),
        HumanMessage(content=(
            f"Research Topic: {topic}\n\n"
            f"=== ArXiv Papers ===\n{paper_context}\n\n"
            f"=== Web Context ===\n{web_context}\n\n"
            f"Write the full literature review now."
        )),
    ]
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def research_gap_stream(topic: str):
    """
    Searches ArXiv and streams a structured research gap analysis.
    """
    llm = get_llm()

    with st.status("🕵️ Scanning research frontier for gaps...", expanded=False) as status:
        papers = search_arxiv(topic, max_results=8)
        status.update(label=f"✅ Analyzed {len(papers)} frontier papers", state="complete")

    if not papers:
        yield "⚠️ No papers found to analyze. Try a different or broader topic."
        return

    st.markdown(f"#### 🕵️ Research Gap Analysis: *{topic}*")
    st.markdown(f"*Based on {len(papers)} recent ArXiv papers*")
    st.divider()

    paper_context = format_papers_for_llm(papers)
    messages = [
        SystemMessage(content=GAP_FINDER_PROMPT),
        HumanMessage(content=(
            f"Research Topic: {topic}\n\n"
            f"=== Available Papers ===\n{paper_context}\n\n"
            f"Identify all significant research gaps and provide your top recommendation."
        )),
    ]
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
