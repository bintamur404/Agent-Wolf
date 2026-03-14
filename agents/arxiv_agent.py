import arxiv
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm import get_llm


SYNTHESIS_PROMPT = """You are Agent Wolf's ArXiv Scholar — a world-class academic research assistant.
You have been given a set of real academic papers from ArXiv on the topic the user asked about.

Your job:
1. Synthesize the key findings from these papers into a clear, insightful response.
2. Highlight the most significant results, methodologies, and trends.
3. Always cite papers using: [Title](arxiv_url) — Author et al. (Year)
4. End with a "Key Takeaways" section with 3-5 bullet points.
5. Be academic but readable. No fluff.

Tone: Expert researcher explaining to a smart colleague.
"""


def search_arxiv(query: str, max_results: int = 5):
    """Searches ArXiv and returns a list of paper metadata dicts."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": ", ".join(a.name for a in result.authors[:3]) + (" et al." if len(result.authors) > 3 else ""),
            "year": result.published.year,
            "abstract": result.summary[:600] + "..." if len(result.summary) > 600 else result.summary,
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "categories": ", ".join(result.categories[:3]),
        })
    return papers


def format_papers_for_llm(papers: list) -> str:
    """Formats paper list into a structured context string for the LLM."""
    if not papers:
        return "No papers found."
    
    formatted = []
    for i, p in enumerate(papers, 1):
        formatted.append(
            f"[Paper {i}]\n"
            f"Title: {p['title']}\n"
            f"Authors: {p['authors']}\n"
            f"Year: {p['year']}\n"
            f"Categories: {p['categories']}\n"
            f"URL: {p['url']}\n"
            f"Abstract: {p['abstract']}\n"
        )
    return "\n---\n".join(formatted)


def arxiv_scholar_stream(query: str):
    """
    Searches ArXiv, displays paper cards in Streamlit, 
    then streams an LLM synthesis of the findings.
    """
    llm = get_llm()

    # Step 1: Search ArXiv
    with st.status("🔬 Searching ArXiv database...", expanded=False) as status:
        papers = search_arxiv(query, max_results=5)
        status.update(
            label=f"✅ Found {len(papers)} papers on ArXiv", state="complete"
        )

    if not papers:
        yield "⚠️ No papers found on ArXiv for this query. Try broader terms."
        return

    # Step 2: Display paper cards inline
    st.markdown("#### 📄 ArXiv Papers Found")
    for i, p in enumerate(papers, 1):
        with st.expander(f"**[{i}] {p['title']}** — {p['authors']} ({p['year']})", expanded=False):
            st.markdown(f"**Categories:** `{p['categories']}`")
            st.markdown(f"**Abstract:** {p['abstract']}")
            col1, col2 = st.columns(2)
            col1.markdown(f"[🔗 ArXiv Page]({p['url']})")
            col2.markdown(f"[📥 PDF]({p['pdf_url']})")

    st.markdown("---")
    st.markdown("#### 🧠 Wolf's Research Synthesis")

    # Step 3: Stream LLM synthesis
    paper_context = format_papers_for_llm(papers)
    messages = [
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(content=(
            f"Research Query: {query}\n\n"
            f"Papers Found:\n{paper_context}\n\n"
            f"Now synthesize the key findings, trends, and insights from these papers."
        )),
    ]

    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
