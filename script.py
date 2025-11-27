# =====================================================
# 🌟 STREAMLIT HYBRID RAG CHATBOT — HF EMBEDDINGS + GEMINI
# =====================================================

import os
import re
import json
import requests
import streamlit as st
from urllib.parse import quote
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun


# =====================================================
# 🌟 STREAMLIT SETUP
# =====================================================
st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")

st.title("🤖 Hybrid RAG Chatbot")
st.write("HuggingFace Embeddings + Gemini + Multi-source Retrieval")


# =====================================================
# 🔑 API KEYS (Streamlit Secrets)
# =====================================================
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]


# GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]


GOOGLE_API_KEY="AIzaSyBMSTBqYv74VqltxMj7G8eUtbuQg8tUROg"
GOOGLE_CSE_ID="94a6404e7eb494900"


if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in secrets")
    st.stop()


# =====================================================
# 🤖 GEMINI LLM
# =====================================================
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=GOOGLE_API_KEY
)


# =====================================================
# 🧠 EMBEDDINGS (HF BGE-SMALL)
# =====================================================
MODEL_NAME = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)


# =====================================================
# 📚 VECTOR DATABASES
# =====================================================
DB1_PATH = r"C:\Users\sadika957\Desktop\small_db_using_HF_baai_bge"
DB2_PATH = r"C:\Users\sadika957\Desktop\large_embeddings_baai_bge"

db1 = Chroma(persist_directory=DB1_PATH, embedding_function=embeddings)
db2 = Chroma(persist_directory=DB2_PATH, embedding_function=embeddings)

retriever1 = db1.as_retriever(search_kwargs={"k": 8})
retriever2 = db2.as_retriever(search_kwargs={"k": 8})


# =====================================================
# 🌐 External Tools
# =====================================================
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
google_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper(
    google_api_key=GOOGLE_API_KEY,
    google_cse_id=GOOGLE_CSE_ID
))


# =====================================================
# 🧩 Helper Functions
# =====================================================
def clean_query(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()


def extractive_answer(query: str, docs: List[Any]) -> str:
    """LLM extracts answer strictly from LOCAL DB context."""
    context_text = "\n\n".join(
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:6])
    )

    prompt = f"""
Answer the question using only the provided CONTEXT.
Each sentence must cite sources like [1], [2], etc.
If not answerable, reply "NOINFO".

Question: {query}

CONTEXT:
{context_text}
"""

    ans = gemini.invoke(prompt).content.strip()
    if ans.upper().startswith("NOINFO") or len(ans) < 30:
        return ""
    return ans


def scholarly_lookup(query: str, limit=3):
    citations = []
    try:
        r = requests.get(
            f"https://api.crossref.org/works?rows={limit}&query={quote(query)}",
            timeout=8
        ).json()
        for item in r.get("message", {}).get("items", []):
            title = item.get("title", ["Untitled"])[0]
            authors = item.get("author", [])
            auth = ", ".join(a.get("family", "") for a in authors[:2]) or "Unknown"
            if len(authors) > 2:
                auth += " et al."
            year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
            doi = item.get("DOI", "")
            link = f"https://doi.org/{doi}" if doi else ""
            citations.append(f"{auth} ({year}). *{title}*. {link}")
        if citations:
            return citations
    except:
        pass
    return ["(No scholarly reference found)"]


def format_clickable_citations(citations: List[str]) -> str:
    md = []
    for i, c in enumerate(citations, 1):
        m = re.search(r"(https?://\S+)", c)
        if m:
            url = m.group(1)
            title = re.search(r"\*([^*]+)\*", c)
            label = title.group(1) if title else f"Source {i}"
            md.append(f"[{i}] [{label}]({url})")
        else:
            md.append(f"[{i}] {c}")
    return "\n".join(md)


# =====================================================
# 📚 Graph State
# =====================================================
class GraphState(TypedDict):
    query: str
    answer: str
    context: str
    citations: List[str]


# =====================================================
# 🧱 Nodes (DB1 → DB2 → Google → Wiki → GBIF → iNat)
# =====================================================
def db1_node(state: GraphState):
    q = clean_query(state["query"])
    docs = retriever1.invoke(q)

    if not docs:
        return {**state, "context": "no_db1"}

    ans = extractive_answer(q, docs)
    if not ans:
        return {**state, "context": "no_db1"}

    refs = [f"https://www.google.com/search?q={quote(q)}"]
    return {**state, "answer": ans, "context": "db1", "citations": refs}


def db2_node(state: GraphState):
    q = clean_query(state["query"])
    docs = retriever2.invoke(q)

    if not docs:
        return {**state, "context": "no_db2"}

    ans = extractive_answer(q, docs)
    if not ans:
        return {**state, "context": "no_db2"}

    refs = scholarly_lookup(q)
    return {**state, "answer": ans, "context": "db2", "citations": refs}


def google_node(state: GraphState):
    q = clean_query(state["query"])
    raw = google_tool.run(q)
    if not raw:
        return {**state, "context": "no_google"}

    ans = gemini.invoke(f"Answer using Google results:\n{raw}").content
    refs = [f"https://www.google.com/search?q={quote(q)}"]
    return {**state, "answer": ans, "context": "google", "citations": refs}


def wiki_node(state: GraphState):
    q = clean_query(state["query"])
    blob = wiki_tool.run(q)
    if not blob:
        return {**state, "context": "no_wiki"}

    ans = gemini.invoke(f"Answer using Wikipedia:\n{blob}").content
    refs = [f"https://en.wikipedia.org/wiki/Special:Search?search={quote(q)}"]
    return {**state, "answer": ans, "context": "wiki", "citations": refs}


def gbif_node(state: GraphState):
    q = clean_query(state["query"])
    try:
        r = requests.get(
            f"https://api.gbif.org/v1/species/search?q={quote(q)}", timeout=8
        ).json()
        species = r.get("results", [])
        if not species:
            return {**state, "context": "no_gbif"}

        lines = [
            f"{sp.get('scientificName')} – https://www.gbif.org/species/{sp.get('key')}"
            for sp in species[:5]
        ]
        ans = "\n".join(lines)
        refs = [f"https://www.gbif.org/species/search?q={quote(q)}"]
        return {**state, "answer": ans, "context": "gbif", "citations": refs}
    except:
        return {**state, "context": "no_gbif"}


def inat_node(state: GraphState):
    q = clean_query(state["query"])
    try:
        r = requests.get(
            f"https://api.inaturalist.org/v1/taxa/autocomplete?q={quote(q)}", timeout=8
        ).json()
        results = r.get("results", [])
        if not results:
            return {**state, "context": "no_inat"}

        lines = [
            f"{it.get('name')} – https://www.inaturalist.org/taxa/{it.get('id')}"
            for it in results[:5]
        ]
        ans = "\n".join(lines)
        refs = [f"https://www.inaturalist.org/search?q={quote(q)}"]
        return {**state, "answer": ans, "context": "inat", "citations": refs}
    except:
        return {**state, "context": "no_inat"}


def final_node(state: GraphState):
    q = clean_query(state["query"])
    base = state["answer"]
    cites = state.get("citations", [])

    prompt = f"""
Summarize the following into a clean, factual answer.
Preserve citations.

Question: {q}
Answer: {base}
"""
    summary = gemini.invoke(prompt).content

    if cites:
        summary += "\n\n### 📚 Citations\n" + format_clickable_citations(cites)

    return {**state, "answer": summary}


# =====================================================
# 🔀 Build Workflow
# =====================================================
workflow = StateGraph(GraphState)

workflow.add_node("db1", db1_node)
workflow.add_node("db2", db2_node)
workflow.add_node("google", google_node)
workflow.add_node("wiki", wiki_node)
workflow.add_node("gbif", gbif_node)
workflow.add_node("inat", inat_node)
workflow.add_node("final", final_node)

workflow.add_edge(START, "db1")
workflow.add_conditional_edges("db1", lambda s: s["context"], {
    "db1": "final",
    "no_db1": "db2"
})
workflow.add_conditional_edges("db2", lambda s: s["context"], {
    "db2": "final",
    "no_db2": "google"
})
workflow.add_conditional_edges("google", lambda s: s["context"], {
    "google": "final",
    "no_google": "wiki"
})
workflow.add_conditional_edges("wiki", lambda s: s["context"], {
    "wiki": "final",
    "no_wiki": "gbif"
})
workflow.add_conditional_edges("gbif", lambda s: s["context"], {
    "gbif": "final",
    "no_gbif": "inat"
})
workflow.add_edge("inat", "final")

graph = workflow.compile()


# =====================================================
# 🌟 STREAMLIT UI
# =====================================================
st.divider()
st.subheader("💬 Ask a question")

user_query = st.text_input("Enter your question:")

if st.button("Run Query"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        state = {
            "query": user_query,
            "answer": "",
            "context": "",
            "citations": []
        }

        with st.spinner("Thinking..."):
            result = graph.invoke(state)

        st.subheader("🤖 Chatbot Response")
        st.markdown(result["answer"], unsafe_allow_html=True)

        st.caption(f"🔍 Source: **{result['context']}**")
