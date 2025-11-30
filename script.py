# ============================================================
# 🌟 STREAMLIT RAG CHATBOT — DB1 + DB2 + GEMINI FLASH
# ============================================================

import streamlit as st
import os
import json
import re
import requests
from urllib.parse import quote
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Google Gemini SDK
import google.generativeai as genai


# ============================================================
# 🔐 ENVIRONMENT VARIABLES
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY environment variable!")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

gemini = genai.GenerativeModel("gemini-2.5-flash")


# ============================================================
# 📁 DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB1_PATH = os.path.join(BASE_DIR, "small_db_using_HF_baai_bge")
DB2_PATH = os.path.join(BASE_DIR, "large_embeddings_baai_bge")


# ============================================================
# 🧠 EMBEDDINGS (HF BGE-SMALL)
# ============================================================

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# ============================================================
# 📚 LOAD VECTOR DBs
# ============================================================

db1 = Chroma(persist_directory=DB1_PATH, embedding_function=embeddings)
retriever1 = db1.as_retriever(search_kwargs={"k": 6})

db2 = Chroma(persist_directory=DB2_PATH, embedding_function=embeddings)
retriever2 = db2.as_retriever(search_kwargs={"k": 6})


# ============================================================
# 🧹 UTILITY FUNCTIONS
# ============================================================

def clean_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip())


def gemini_answer(prompt: str) -> str:
    """Generate text from Gemini with correct API."""
    try:
        resp = gemini.generate_content(prompt)
        return resp.text.strip() if resp.text else ""
    except Exception as e:
        return f"(Gemini error: {e})"


def extractive_answer(query: str, docs: List[Any]) -> str:
    """Extractive QA: answer ONLY from context."""
    context_text = "\n\n".join(
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:5])
    )

    prompt = f"""
Answer the question ONLY using the provided CONTEXT.
Cite sources using [1], [2], etc.
If answer not found, return "NOINFO".

Question: {query}

CONTEXT:
{context_text}
"""

    ans = gemini_answer(prompt)

    if ans.upper().startswith("NOINFO") or len(ans) < 20:
        return ""

    return ans


# ============================================================
# 📚 GRAPH STATE
# ============================================================

class GraphState(TypedDict):
    query: str
    answer: str
    context: str


# ============================================================
# 🧱 GRAPH NODES
# ============================================================

def db1_node(state: GraphState):
    q = clean_query(state["query"])
    docs = retriever1.invoke(q)

    if not docs:
        return {**state, "context": "no_db1"}

    ans = extractive_answer(q, docs)
    if not ans:
        return {**state, "context": "no_db1"}

    return {**state, "answer": ans, "context": "db1"}


def db2_node(state: GraphState):
    q = clean_query(state["query"])
    docs = retriever2.invoke(q)

    if not docs:
        return {**state, "context": "no_db2"}

    ans = extractive_answer(q, docs)
    if not ans:
        return {**state, "context": "no_db2"}

    return {**state, "answer": ans, "context": "db2"}


# def google_node(state: GraphState):
#     q = clean_query(state["query"])
#     try:
#         snippet = requests.get(
#             f"https://www.googleapis.com/customsearch/v1?q={quote(q)}&key={GOOGLE_API_KEY}",
#             timeout=8,
#         ).json()

#         text = snippet.get("items", [{}])[0].get("snippet", "")
#         if not text:
#             return {**state, "context": "no_google"}

#         ans = gemini_answer(f"Answer concisely using Google results:\n{text}")
#         return {**state, "answer": ans, "context": "google"}

#     except:
#         return {**state, "context": "no_google"}

def google_node(state: GraphState):
    q = clean_query(state["query"])
    try:
        snippet = google_tool.run(q)

        # Some APIs return [] or {} = treat as no results
        if not snippet or len(str(snippet).strip()) < 10:
            return {**state, "context": "no_google"}

        # Proper Gemini answer
        resp = gemini.generate_content(
            f"Answer using these Google search results:\n{snippet}"
        )
        ans = resp.text.strip()

        return {
            **state,
            "answer": ans,
            "context": "google",
            "citations": [
                f"https://www.google.com/search?q={quote(q)}"
            ]
        }

    except Exception as e:
        print("Google error:", e)
        return {**state, "context": "no_google"}



def wiki_node(state: GraphState):
    q = clean_query(state["query"])
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(q)}"
        blob = requests.get(url, timeout=8).json().get("extract", "")

        if not blob:
            return {**state, "context": "no_wiki"}

        ans = gemini_answer(f"Answer using this Wikipedia extract:\n{blob}")
        return {**state, "answer": ans, "context": "wiki"}

    except:
        return {**state, "context": "no_wiki"}


def final_node(state: GraphState):
    """Final cleanup summary."""
    q = clean_query(state["query"])
    base_ans = state["answer"]

    prompt = f"""
Rewrite the following answer more clearly and concisely.

Question: {q}
Answer: {base_ans}
"""

    summary = gemini_answer(prompt)
    return {**state, "answer": summary}


# ============================================================
# 🔀 GRAPH PIPELINE
# ============================================================

workflow = StateGraph(GraphState)

workflow.add_node("db1", db1_node)
workflow.add_node("db2", db2_node)
workflow.add_node("google", google_node)
workflow.add_node("wiki", wiki_node)
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

workflow.add_edge("wiki", "final")
workflow.add_edge("final", END)

graph = workflow.compile()


# ============================================================
# 🎨 STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")
st.title("🤖 Hybrid RAG Chatbot — DB1 + DB2 + Google + Wiki")


# Initialize conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Display conversation
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# User input box
user_query = st.chat_input("Ask something...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.write(user_query)

    # Run RAG pipeline
    result = graph.invoke({
        "query": user_query,
        "context": "",
        "answer": ""
    })

    answer = result["answer"]
    context = result["context"]

    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"Source: {context}")

    st.session_state["messages"].append({"role": "assistant", "content": answer})










# import streamlit as st
# import os
# import re
# from typing import TypedDict, List, Any
# from langgraph.graph import StateGraph, START, END

# # from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from google.generativeai import configure, GenerativeModel


# # =============================================
# # 🔑 Environment Variables
# # =============================================
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# if not GOOGLE_API_KEY:
#     st.error("❌ Missing GOOGLE_API_KEY environment variable")
#     st.stop()


# # # =============================================
# # # 🤖 Gemini LLM
# # # =============================================
# # gemini = ChatGoogleGenerativeAI(
# #     model="gemini-2.5-flash",
# #     api_key=GOOGLE_API_KEY,
# #     temperature=0
# # )


# # =====================================================
# # 🤖 GEMINI LLM
# # =====================================================
# configure(api_key=GOOGLE_API_KEY)
# gemini = GenerativeModel("gemini-2.5-flash")


# # =============================================
# # 🧠 Embedding Model (CPU-friendly)
# # =============================================
# MODEL_NAME = "BAAI/bge-small-en-v1.5"

# embeddings = HuggingFaceEmbeddings(
#     model_name=MODEL_NAME,
#     model_kwargs={"device": "cpu"}  # DigitalOcean safe
# )


# # =============================================
# # 📚 Vector DBs
# # =============================================
# DB1_PATH = os.path.join(BASE_DIR, "small_db_using_HF_baai_bge")
# DB2_PATH = os.path.join(BASE_DIR, "large_embeddings_baai_bge")

# db1 = Chroma(persist_directory=DB1_PATH, embedding_function=embeddings)
# db2 = Chroma(persist_directory=DB2_PATH, embedding_function=embeddings)

# retriever1 = db1.as_retriever(search_kwargs={"k": 8})
# retriever2 = db2.as_retriever(search_kwargs={"k": 8})


# # =============================================
# # 🔧 Utility Functions
# # =============================================
# def clean_query(q: str):
#     return re.sub(r"\s+", " ", q).strip()


# def extractive_answer(query: str, docs: List[Any]) -> str:
#     """Use Gemini to answer strictly from DB context."""
#     context = "\n\n".join(d.page_content for d in docs[:6])

#     prompt = f"""
# Use ONLY the following context to answer.
# If answer not found, reply "NOINFO".

# QUESTION:
# {query}

# CONTEXT:
# {context}
# """

#     out = gemini.invoke(prompt).content.strip()

#     if out.upper().startswith("NOINFO") or len(out) < 10:
#         return ""
#     return out


# # =============================================
# # 🧩 Graph State
# # =============================================
# class GraphState(TypedDict):
#     query: str
#     answer: str
#     context: str


# # =============================================
# # 🔰 Node Functions
# # =============================================
# def db1_node(state: GraphState):
#     q = clean_query(state["query"])
#     docs = retriever1.invoke(q)
    
#     if not docs:
#         return {**state, "context": "no_db1"}

#     ans = extractive_answer(q, docs)
#     if not ans:
#         return {**state, "context": "no_db1"}

#     return {**state, "answer": ans, "context": "db1"}


# def db2_node(state: GraphState):
#     q = clean_query(state["query"])
#     docs = retriever2.invoke(q)

#     if not docs:
#         return {**state, "context": "no_db2"}

#     ans = extractive_answer(q, docs)
#     if not ans:
#         return {**state, "context": "no_db2"}

#     return {**state, "answer": ans, "context": "db2"}


# def final_node(state: GraphState):
#     """Final Gemini summary."""
#     q = clean_query(state["query"])
#     base = state["answer"]

#     summary = gemini.invoke(f"Summarize clearly:\n\n{base}").content.strip()
    
#     return {**state, "answer": summary}


# # =============================================
# # 🔗 Graph Workflow
# # =============================================
# workflow = StateGraph(GraphState)

# workflow.add_node("db1", db1_node)
# workflow.add_node("db2", db2_node)
# workflow.add_node("final", final_node)

# workflow.add_edge(START, "db1")

# workflow.add_conditional_edges("db1", lambda s: s["context"], {
#     "db1": "final",
#     "no_db1": "db2"
# })

# workflow.add_conditional_edges("db2", lambda s: s["context"], {
#     "db2": "final",
#     "no_db2": "final"
# })

# workflow.add_edge("final", END)

# graph = workflow.compile()


# # =============================================
# # 🎨 Streamlit UI
# # =============================================
# st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")
# st.title("🤖 Hybrid RAG Chatbot — v1")
# st.write("DB1 → DB2 → Gemini")

# query = st.text_input("Ask your question:")

# if st.button("Search"):
#     if not query.strip():
#         st.warning("Please enter a question.")
#         st.stop()

#     with st.spinner("Thinking..."):
#         result = graph.invoke({
#             "query": query,
#             "context": "",
#             "answer": ""
#         })

#     st.subheader("Response")
#     st.write(result["answer"])

#     st.caption(f"Source: **{result['context']}**")























# # NEW CODE TO RESOLVE ERROR:
# # AttributeError: 'GenerativeModel' object has no attribute 'invoke'

# # [NOTE] During task with name 'google' and id 

# # '6e717481-f8c5-5e21-04d3-a249d9c12959'
# # =====================================================
# # 🌟 STREAMLIT HYBRID RAG CHATBOT — HF EMBEDDINGS + GEMINI
# # =====================================================

# import os
# import re
# import json
# import requests
# import streamlit as st
# from urllib.parse import quote
# from typing import TypedDict, List, Dict, Any

# from langgraph.graph import StateGraph, START, END

# # New embeddings + chroma + google search packages
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_community.tools import WikipediaQueryRun
# from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun

# from google.generativeai import configure, GenerativeModel


# # =====================================================
# # 🌟 STREAMLIT SETUP
# # =====================================================
# st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")

# st.title("🤖 Hybrid RAG Chatbot")
# st.write("HuggingFace Embeddings + Gemini + Multi-source Retrieval")


# # =====================================================
# # 🔑 API KEYS (Streamlit Secrets)
# # =====================================================
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
# GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]

# if not GOOGLE_API_KEY:
#     st.error("❌ GOOGLE_API_KEY not found in secrets")
#     st.stop()


# # =====================================================
# # 🤖 GEMINI LLM
# # =====================================================
# configure(api_key=GOOGLE_API_KEY)
# gemini = GenerativeModel("gemini-2.5-flash")


# # =====================================================
# # 🧠 EMBEDDINGS (HF BGE-SMALL)
# # =====================================================
# MODEL_NAME = "BAAI/bge-small-en-v1.5"
# embeddings = HuggingFaceEmbeddings(
#     model_name=MODEL_NAME,
#     # Force CPU so it works on Streamlit Cloud / CPU-only environments
#     model_kwargs={"device": "cpu"},
# )


# # =====================================================
# # 📚 VECTOR DATABASES
# # =====================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Use relative paths so it works on Linux/Streamlit/DigitalOcean
# DB1_PATH = os.path.join(BASE_DIR, "small_db_using_HF_baai_bge")
# DB2_PATH = os.path.join(BASE_DIR, "large_embeddings_baai_bge")

# db1 = Chroma(persist_directory=DB1_PATH, embedding_function=embeddings)
# db2 = Chroma(persist_directory=DB2_PATH, embedding_function=embeddings)

# retriever1 = db1.as_retriever(search_kwargs={"k": 8})
# retriever2 = db2.as_retriever(search_kwargs={"k": 8})


# # =====================================================
# # 🌐 External Tools
# # =====================================================
# wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# google_tool = GoogleSearchRun(
#     api_wrapper=GoogleSearchAPIWrapper(
#         google_api_key=GOOGLE_API_KEY,
#         google_cse_id=GOOGLE_CSE_ID,
#     )
# )


# # =====================================================
# # 🧩 Helper Functions
# # =====================================================
# def clean_query(q: str) -> str:
#     return re.sub(r"\s+", " ", q).strip()


# def extractive_answer(query: str, docs: List[Any]) -> str:
#     """LLM extracts answer strictly from LOCAL DB context."""
#     context_text = "\n\n".join(
#         f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:6])
#     )

#     prompt = f"""
# Answer the question using only the provided CONTEXT.
# Each sentence must cite sources like [1], [2], etc.
# If not answerable, reply "NOINFO".

# Question: {query}

# CONTEXT:
# {context_text}
# """

#     try:
#         response = gemini.generate_content(prompt)
#         ans = (response.text or "").strip()
#     except Exception:
#         return ""

#     if ans.upper().startswith("NOINFO") or len(ans) < 30:
#         return ""
#     return ans


# def scholarly_lookup(query: str, limit=3):
#     citations = []
#     try:
#         r = requests.get(
#             f"https://api.crossref.org/works?rows={limit}&query={quote(query)}",
#             timeout=8,
#         ).json()
#         for item in r.get("message", {}).get("items", []):
#             title = item.get("title", ["Untitled"])[0]
#             authors = item.get("author", [])
#             auth = ", ".join(a.get("family", "") for a in authors[:2]) or "Unknown"
#             if len(authors) > 2:
#                 auth += " et al."
#             year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
#             doi = item.get("DOI", "")
#             link = f"https://doi.org/{doi}" if doi else ""
#             citations.append(f"{auth} ({year}). *{title}*. {link}")
#         if citations:
#             return citations
#     except Exception:
#         pass
#     return ["(No scholarly reference found)"]


# def format_clickable_citations(citations: List[str]) -> str:
#     md = []
#     for i, c in enumerate(citations, 1):
#         m = re.search(r"(https?://\S+)", c)
#         if m:
#             url = m.group(1)
#             title = re.search(r"\*([^*]+)\*", c)
#             label = title.group(1) if title else f"Source {i}"
#             md.append(f"[{i}] [{label}]({url})")
#         else:
#             md.append(f"[{i}] {c}")
#     return "\n".join(md)


# # =====================================================
# # 📚 Graph State
# # =====================================================
# class GraphState(TypedDict):
#     query: str
#     answer: str
#     context: str
#     citations: List[str]


# # # =====================================================
# # # 🧱 Nodes (DB1 → DB2 → Google → Wiki → GBIF → iNat)
# # # =====================================================
# # def db1_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     docs = retriever1.invoke(q)

# #     if not docs:
# #         return {**state, "context": "no_db1"}

# #     ans = extractive_answer(q, docs)
# #     if not ans:
# #         return {**state, "context": "no_db1"}

# #     refs = [f"https://www.google.com/search?q={quote(q)}"]
# #     return {**state, "answer": ans, "context": "db1", "citations": refs}


# # def db2_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     docs = retriever2.invoke(q)

# #     if not docs:
# #         return {**state, "context": "no_db2"}

# #     ans = extractive_answer(q, docs)
# #     if not ans:
# #         return {**state, "context": "no_db2"}

# #     refs = scholarly_lookup(q)
# #     return {**state, "answer": ans, "context": "db2", "citations": refs}



# # =====================================================
# # 🧱 Nodes (DB1 → DB2 → Gemini → Final)
# # =====================================================

# def db1_node(state: GraphState):
#     q = clean_query(state["query"])
#     docs = retriever1.invoke(q)

#     if not docs:
#         return {**state, "context": "no_db1"}

#     ans = extractive_answer(q, docs)
#     if not ans:
#         return {**state, "context": "no_db1"}

#     # DB1 citations
#     refs = [f"LocalDB1: {len(docs)} matches"]
#     return {**state, "answer": ans, "context": "db1", "citations": refs}


# def db2_node(state: GraphState):
#     q = clean_query(state["query"])
#     docs = retriever2.invoke(q)

#     if not docs:
#         return {**state, "context": "no_db2"}

#     ans = extractive_answer(q, docs)
#     if not ans:
#         return {**state, "context": "no_db2"}

#     # DB2 citations
#     refs = [f"LocalDB2: {len(docs)} matches"]
#     return {**state, "answer": ans, "context": "db2", "citations": refs}


# # GEMINI FALLBACK
# def gemini_node(state: GraphState):
#     q = clean_query(state["query"])

#     try:
#         prompt = f"""
# You are a helpful assistant. The local database could not answer the question.
# Give a clear factual answer.

# Question: {q}
# """
#         response = gemini.generate_content(prompt)
#         ans = response.text or ""
#     except Exception:
#         return {**state, "context": "no_gemini"}

#     refs = ["Gemini Fallback"]
#     return {**state, "answer": ans, "context": "gemini", "citations": refs}


# # FINAL NODE
# def final_node(state: GraphState):
#     q = clean_query(state["query"])
#     base = state["answer"]
#     cites = state.get("citations", [])

#     try:
#         prompt = f"""
# Summarize the following into a clean, high-quality answer.
# Preserve citations.

# Question: {q}
# Answer: {base}
# """
#         response = gemini.generate_content(prompt)
#         summary = response.text or base
#     except Exception:
#         summary = base

#     if cites:
#         summary += "\n\n### 📚 Citations\n" + "\n".join(f"- {c}" for c in cites)

#     return {**state, "answer": summary}



# def google_node(state: GraphState):
#     q = clean_query(state["query"])
#     raw = google_tool.run(q)
#     if not raw:
#         return {**state, "context": "no_google"}

#     try:
#         response = gemini.generate_content(f"Answer using Google results:\n{raw}")
#         ans = response.text or ""
#     except Exception:
#         return {**state, "context": "no_google"}

#     refs = [f"https://www.google.com/search?q={quote(q)}"]
#     return {**state, "answer": ans, "context": "google", "citations": refs}


# def wiki_node(state: GraphState):
#     q = clean_query(state["query"])
#     blob = wiki_tool.run(q)
#     if not blob:
#         return {**state, "context": "no_wiki"}

#     try:
#         response = gemini.generate_content(f"Answer using Wikipedia:\n{blob}")
#         ans = response.text or ""
#     except Exception:
#         return {**state, "context": "no_wiki"}

#     refs = [f"https://en.wikipedia.org/wiki/Special:Search?search={quote(q)}"]
#     return {**state, "answer": ans, "context": "wiki", "citations": refs}


# def gbif_node(state: GraphState):
#     q = clean_query(state["query"])
#     try:
#         r = requests.get(
#             f"https://api.gbif.org/v1/species/search?q={quote(q)}", timeout=8
#         ).json()
#         species = r.get("results", [])
#         if not species:
#             return {**state, "context": "no_gbif"}

#         lines = [
#             f"{sp.get('scientificName')} – https://www.gbif.org/species/{sp.get('key')}"
#             for sp in species[:5]
#         ]
#         ans = "\n".join(lines)
#         refs = [f"https://www.gbif.org/species/search?q={quote(q)}"]
#         return {**state, "answer": ans, "context": "gbif", "citations": refs}
#     except Exception:
#         return {**state, "context": "no_gbif"}


# def inat_node(state: GraphState):
#     q = clean_query(state["query"])
#     try:
#         r = requests.get(
#             f"https://api.inaturalist.org/v1/taxa/autocomplete?q={quote(q)}",
#             timeout=8,
#         ).json()
#         results = r.get("results", [])
#         if not results:
#             return {**state, "context": "no_inat"}

#         lines = [
#             f"{it.get('name')} – https://www.inaturalist.org/taxa/{it.get('id')}"
#             for it in results[:5]
#         ]
#         ans = "\n".join(lines)
#         refs = [f"https://www.inaturalist.org/search?q={quote(q)}"]
#         return {**state, "answer": ans, "context": "inat", "citations": refs}
#     except Exception:
#         return {**state, "context": "no_inat"}


# def final_node(state: GraphState):
#     q = clean_query(state["query"])
#     base = state["answer"]
#     cites = state.get("citations", [])

#     prompt = f"""
# Summarize the following into a clean, factual answer.
# Preserve citations.

# Question: {q}
# Answer: {base}
# """
#     try:
#         response = gemini.generate_content(prompt)
#         summary = response.text or ""
#     except Exception:
#         summary = base

#     if cites:
#         summary += "\n\n### 📚 Citations\n" + format_clickable_citations(cites)

#     return {**state, "answer": summary}


# # # =====================================================
# # # 🔀 Build Workflow
# # # =====================================================
# # workflow = StateGraph(GraphState)

# # workflow.add_node("db1", db1_node)
# # workflow.add_node("db2", db2_node)
# # workflow.add_node("google", google_node)
# # workflow.add_node("wiki", wiki_node)
# # workflow.add_node("gbif", gbif_node)
# # workflow.add_node("inat", inat_node)
# # workflow.add_node("final", final_node)

# # workflow.add_edge(START, "db1")
# # workflow.add_conditional_edges("db1", lambda s: s["context"], {
# #     "db1": "final",
# #     "no_db1": "db2",
# # })
# # workflow.add_conditional_edges("db2", lambda s: s["context"], {
# #     "db2": "final",
# #     "no_db2": "google",
# # })
# # workflow.add_conditional_edges("google", lambda s: s["context"], {
# #     "google": "final",
# #     "no_google": "wiki",
# # })
# # workflow.add_conditional_edges("wiki", lambda s: s["context"], {
# #     "wiki": "final",
# #     "no_wiki": "gbif",
# # })
# # workflow.add_conditional_edges("gbif", lambda s: s["context"], {
# #     "gbif": "final",
# #     "no_gbif": "inat",
# # })
# # workflow.add_edge("inat", "final")

# # graph = workflow.compile()





# # =====================================================
# # 🔀 Build Workflow (DB1 → DB2 → Gemini → Final)
# # =====================================================

# workflow = StateGraph(GraphState)

# workflow.add_node("db1", db1_node)
# workflow.add_node("db2", db2_node)
# workflow.add_node("gemini", gemini_node)
# workflow.add_node("final", final_node)

# workflow.add_edge(START, "db1")

# # DB1
# workflow.add_conditional_edges("db1", lambda s: s["context"], {
#     "db1": "final",
#     "no_db1": "db2",
# })

# # DB2
# workflow.add_conditional_edges("db2", lambda s: s["context"], {
#     "db2": "final",
#     "no_db2": "gemini",
# })

# # Gemini fallback always goes to final
# workflow.add_edge("gemini", "final")

# graph = workflow.compile()



# # =====================================================
# # 🌟 STREAMLIT UI
# # =====================================================
# st.divider()
# st.subheader("💬 Ask a question")

# user_query = st.text_input("Enter your question:")

# if st.button("Run Query"):
#     if not user_query.strip():
#         st.warning("Please enter a question.")
#     else:
#         state: GraphState = {
#             "query": user_query,
#             "answer": "",
#             "context": "",
#             "citations": [],
#         }

#         with st.spinner("Thinking..."):
#             result = graph.invoke(state)

#         st.subheader("🤖 Chatbot Response")
#         st.markdown(result["answer"], unsafe_allow_html=True)

#         st.caption(f"🔍 Source: **{result['context']}**")













# # # =====================================================
# # # 🌟 STREAMLIT HYBRID RAG CHATBOT — HF EMBEDDINGS + GEMINI
# # # =====================================================

# # import os
# # import re
# # import json
# # import requests
# # import streamlit as st
# # from urllib.parse import quote
# # from typing import TypedDict, List, Dict, Any

# # from langgraph.graph import StateGraph, START, END
# # # from langchain_google_genai import ChatGoogleGenerativeAI

# # # from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain_huggingface import HuggingFaceEmbeddings

# # from langchain_community.vectorstores import Chroma

# # from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
# # from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun

# # from google.generativeai import configure, GenerativeModel

# # # =====================================================
# # # 🌟 STREAMLIT SETUP
# # # =====================================================
# # st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")

# # st.title("🤖 Hybrid RAG Chatbot")
# # st.write("HuggingFace Embeddings + Gemini + Multi-source Retrieval")


# # # =====================================================
# # # 🔑 API KEYS (Streamlit Secrets)
# # # =====================================================
# # GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]


# # GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]



# # if not GOOGLE_API_KEY:
# #     st.error("❌ GOOGLE_API_KEY not found in secrets")
# #     st.stop()


# # # =====================================================
# # # 🤖 GEMINI LLM
# # # =====================================================
# # # gemini = ChatGoogleGenerativeAI(
# # #     model="gemini-2.5-flash",
# # #     temperature=0,
# # #     api_key=GOOGLE_API_KEY
# # # )

# # configure(api_key=GOOGLE_API_KEY)
# # gemini = GenerativeModel("gemini-2.5-flash")


# # # =====================================================
# # # 🧠 EMBEDDINGS (HF BGE-SMALL)
# # # =====================================================
# # MODEL_NAME = "BAAI/bge-small-en-v1.5"
# # embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)


# # # =====================================================
# # # 📚 VECTOR DATABASES
# # # =====================================================
# # DB1_PATH = r"C:\Users\sadika957\Desktop\small_db_using_HF_baai_bge"
# # DB2_PATH = r"C:\Users\sadika957\Desktop\large_embeddings_baai_bge"

# # db1 = Chroma(persist_directory=DB1_PATH, embedding_function=embeddings)
# # db2 = Chroma(persist_directory=DB2_PATH, embedding_function=embeddings)

# # retriever1 = db1.as_retriever(search_kwargs={"k": 8})
# # retriever2 = db2.as_retriever(search_kwargs={"k": 8})


# # # =====================================================
# # # 🌐 External Tools
# # # =====================================================
# # wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# # google_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper(
# #     google_api_key=GOOGLE_API_KEY,
# #     google_cse_id=GOOGLE_CSE_ID
# # ))


# # # =====================================================
# # # 🧩 Helper Functions
# # # =====================================================
# # def clean_query(q: str) -> str:
# #     return re.sub(r"\s+", " ", q).strip()


# # def extractive_answer(query: str, docs: List[Any]) -> str:
# #     """LLM extracts answer strictly from LOCAL DB context."""
# #     context_text = "\n\n".join(
# #         f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:6])
# #     )

# #     prompt = f"""
# # Answer the question using only the provided CONTEXT.
# # Each sentence must cite sources like [1], [2], etc.
# # If not answerable, reply "NOINFO".

# # Question: {query}

# # CONTEXT:
# # {context_text}
# # """

# #     ans = gemini.invoke(prompt).content.strip()
# #     if ans.upper().startswith("NOINFO") or len(ans) < 30:
# #         return ""
# #     return ans


# # def scholarly_lookup(query: str, limit=3):
# #     citations = []
# #     try:
# #         r = requests.get(
# #             f"https://api.crossref.org/works?rows={limit}&query={quote(query)}",
# #             timeout=8
# #         ).json()
# #         for item in r.get("message", {}).get("items", []):
# #             title = item.get("title", ["Untitled"])[0]
# #             authors = item.get("author", [])
# #             auth = ", ".join(a.get("family", "") for a in authors[:2]) or "Unknown"
# #             if len(authors) > 2:
# #                 auth += " et al."
# #             year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
# #             doi = item.get("DOI", "")
# #             link = f"https://doi.org/{doi}" if doi else ""
# #             citations.append(f"{auth} ({year}). *{title}*. {link}")
# #         if citations:
# #             return citations
# #     except:
# #         pass
# #     return ["(No scholarly reference found)"]


# # def format_clickable_citations(citations: List[str]) -> str:
# #     md = []
# #     for i, c in enumerate(citations, 1):
# #         m = re.search(r"(https?://\S+)", c)
# #         if m:
# #             url = m.group(1)
# #             title = re.search(r"\*([^*]+)\*", c)
# #             label = title.group(1) if title else f"Source {i}"
# #             md.append(f"[{i}] [{label}]({url})")
# #         else:
# #             md.append(f"[{i}] {c}")
# #     return "\n".join(md)


# # # =====================================================
# # # 📚 Graph State
# # # =====================================================
# # class GraphState(TypedDict):
# #     query: str
# #     answer: str
# #     context: str
# #     citations: List[str]


# # # =====================================================
# # # 🧱 Nodes (DB1 → DB2 → Google → Wiki → GBIF → iNat)
# # # =====================================================
# # def db1_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     docs = retriever1.invoke(q)

# #     if not docs:
# #         return {**state, "context": "no_db1"}

# #     ans = extractive_answer(q, docs)
# #     if not ans:
# #         return {**state, "context": "no_db1"}

# #     refs = [f"https://www.google.com/search?q={quote(q)}"]
# #     return {**state, "answer": ans, "context": "db1", "citations": refs}


# # def db2_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     docs = retriever2.invoke(q)

# #     if not docs:
# #         return {**state, "context": "no_db2"}

# #     ans = extractive_answer(q, docs)
# #     if not ans:
# #         return {**state, "context": "no_db2"}

# #     refs = scholarly_lookup(q)
# #     return {**state, "answer": ans, "context": "db2", "citations": refs}


# # def google_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     raw = google_tool.run(q)
# #     if not raw:
# #         return {**state, "context": "no_google"}

# #     ans = gemini.invoke(f"Answer using Google results:\n{raw}").content
# #     refs = [f"https://www.google.com/search?q={quote(q)}"]
# #     return {**state, "answer": ans, "context": "google", "citations": refs}


# # def wiki_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     blob = wiki_tool.run(q)
# #     if not blob:
# #         return {**state, "context": "no_wiki"}

# #     ans = gemini.invoke(f"Answer using Wikipedia:\n{blob}").content
# #     refs = [f"https://en.wikipedia.org/wiki/Special:Search?search={quote(q)}"]
# #     return {**state, "answer": ans, "context": "wiki", "citations": refs}


# # def gbif_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     try:
# #         r = requests.get(
# #             f"https://api.gbif.org/v1/species/search?q={quote(q)}", timeout=8
# #         ).json()
# #         species = r.get("results", [])
# #         if not species:
# #             return {**state, "context": "no_gbif"}

# #         lines = [
# #             f"{sp.get('scientificName')} – https://www.gbif.org/species/{sp.get('key')}"
# #             for sp in species[:5]
# #         ]
# #         ans = "\n".join(lines)
# #         refs = [f"https://www.gbif.org/species/search?q={quote(q)}"]
# #         return {**state, "answer": ans, "context": "gbif", "citations": refs}
# #     except:
# #         return {**state, "context": "no_gbif"}


# # def inat_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     try:
# #         r = requests.get(
# #             f"https://api.inaturalist.org/v1/taxa/autocomplete?q={quote(q)}", timeout=8
# #         ).json()
# #         results = r.get("results", [])
# #         if not results:
# #             return {**state, "context": "no_inat"}

# #         lines = [
# #             f"{it.get('name')} – https://www.inaturalist.org/taxa/{it.get('id')}"
# #             for it in results[:5]
# #         ]
# #         ans = "\n".join(lines)
# #         refs = [f"https://www.inaturalist.org/search?q={quote(q)}"]
# #         return {**state, "answer": ans, "context": "inat", "citations": refs}
# #     except:
# #         return {**state, "context": "no_inat"}


# # def final_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     base = state["answer"]
# #     cites = state.get("citations", [])

# #     prompt = f"""
# # Summarize the following into a clean, factual answer.
# # Preserve citations.

# # Question: {q}
# # Answer: {base}
# # """
# #     summary = gemini.invoke(prompt).content

# #     if cites:
# #         summary += "\n\n### 📚 Citations\n" + format_clickable_citations(cites)

# #     return {**state, "answer": summary}


# # # =====================================================
# # # 🔀 Build Workflow
# # # =====================================================
# # workflow = StateGraph(GraphState)

# # workflow.add_node("db1", db1_node)
# # workflow.add_node("db2", db2_node)
# # workflow.add_node("google", google_node)
# # workflow.add_node("wiki", wiki_node)
# # workflow.add_node("gbif", gbif_node)
# # workflow.add_node("inat", inat_node)
# # workflow.add_node("final", final_node)

# # workflow.add_edge(START, "db1")
# # workflow.add_conditional_edges("db1", lambda s: s["context"], {
# #     "db1": "final",
# #     "no_db1": "db2"
# # })
# # workflow.add_conditional_edges("db2", lambda s: s["context"], {
# #     "db2": "final",
# #     "no_db2": "google"
# # })
# # workflow.add_conditional_edges("google", lambda s: s["context"], {
# #     "google": "final",
# #     "no_google": "wiki"
# # })
# # workflow.add_conditional_edges("wiki", lambda s: s["context"], {
# #     "wiki": "final",
# #     "no_wiki": "gbif"
# # })
# # workflow.add_conditional_edges("gbif", lambda s: s["context"], {
# #     "gbif": "final",
# #     "no_gbif": "inat"
# # })
# # workflow.add_edge("inat", "final")

# # graph = workflow.compile()


# # # =====================================================
# # # 🌟 STREAMLIT UI
# # # =====================================================
# # st.divider()
# # st.subheader("💬 Ask a question")

# # user_query = st.text_input("Enter your question:")

# # if st.button("Run Query"):
# #     if not user_query.strip():
# #         st.warning("Please enter a question.")
# #     else:
# #         state = {
# #             "query": user_query,
# #             "answer": "",
# #             "context": "",
# #             "citations": []
# #         }

# #         with st.spinner("Thinking..."):
# #             result = graph.invoke(state)

# #         st.subheader("🤖 Chatbot Response")
# #         st.markdown(result["answer"], unsafe_allow_html=True)

# #         st.caption(f"🔍 Source: **{result['context']}**")
