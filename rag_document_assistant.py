"""
Assignment 1.2: RAG-Based Intelligent Document Assistant (LangChain + LlamaIndex)

Deployable Streamlit app. Steps 1-14 contain exactly the same code as RAG_Document_Assistant.ipynb;
the Streamlit UI is added at the end.

Run:  streamlit run rag_document_assistant.py

LlamaIndex -> ingestion, chunking, embeddings, vector index, retriever
LangChain  -> LLM, prompt templates, LCEL chains / runnables, output parsing
"""


# ======================================================================================
# Step 1: Import libraries and load environment variables
# ======================================================================================
import os
import json
import hashlib
import warnings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Literal, Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Supressing warnings for the sake of a clean output
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Which LLM provider to use: "groq" (default) or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()


def missing_api_key() -> Optional[str]:
    """Return the name of the missing API key, or None if the key is present."""
    key_name = "OPENAI_API_KEY" if LLM_PROVIDER == "openai" else "GROQ_API_KEY"
    return None if os.getenv(key_name) else key_name


# ======================================================================================
# Step 2: Configuration
# ======================================================================================
# Folder paths
DATA_DIR = "data"          # put your PDFs / TXT / MD / DOCX files here
STORAGE_DIR = "storage"    # LlamaIndex will save the vector index here
URLS_FILE = os.path.join(STORAGE_DIR, "web_urls.json")
FINGERPRINT_FILE = os.path.join(STORAGE_DIR, "fingerprint.json")

# Supported file types
SUPPORTED_EXTS = [".pdf", ".txt", ".md", ".docx"]

# Chunking + retrieval settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 4
MAX_CHARS_PER_CALL = 6000    # summarization: text sent per LLM call (keeps each request small)

# Groq free tier allows only a few thousand tokens per minute (TPM).
# These settings keep requests small and wait + retry when the limit is hit.
MAX_OUTPUT_TOKENS = 2048      # cap on the length of each LLM answer
RATE_LIMIT_RETRIES = 4        # how many times to retry after a rate-limit error
RATE_LIMIT_WAIT_SECONDS = 30  # wait time before retrying

# Models
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_VERSION = 2   # change this number to force every saved index to be rebuilt
GROQ_MODEL = "openai/gpt-oss-20b"
OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)


# Web pages to ingest (stored in a small JSON file)
def load_saved_urls() -> List[str]:
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE) as f:
            return json.load(f)
    return []


def save_urls(urls: List[str]):
    with open(URLS_FILE, "w") as f:
        json.dump(urls, f)


def add_url(url: str):
    urls = load_saved_urls()
    if url.strip() and url.strip() not in urls:
        save_urls(urls + [url.strip()])


# ======================================================================================
# Step 3: LLM integration using LangChain
# ======================================================================================
def get_llm():
    """Return a LangChain chat model based on LLM_PROVIDER."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS)

    from langchain_groq import ChatGroq
    return ChatGroq(model=GROQ_MODEL, temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS)


def is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return "rate_limit" in text or "rate limit" in text or "429" in text or "413" in text


def invoke_with_retry(chain, inputs: dict):
    """Invoke a LangChain chain; if the provider's rate limit is hit, wait and try again."""
    import time
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if not is_rate_limit_error(e) or attempt == RATE_LIMIT_RETRIES:
                raise
            print(f"Rate limit reached - waiting {RATE_LIMIT_WAIT_SECONDS}s before retry "
                  f"({attempt + 1}/{RATE_LIMIT_RETRIES})...")
            time.sleep(RATE_LIMIT_WAIT_SECONDS)


# ======================================================================================
# Step 4: Configure LlamaIndex (embeddings + chunking)
# ======================================================================================
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter


def configure_llamaindex():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Embedding model (runs locally, free)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    # Chunking strategy
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # LLM is handled by LangChain, so we switch off LlamaIndex's own LLM
    Settings.llm = None


# ======================================================================================
# Step 5: Document ingestion (local files + web pages)
# ======================================================================================
from llama_index.core import SimpleDirectoryReader, Document


def load_local_documents(data_dir: str = DATA_DIR) -> List[Document]:
    """Load all supported files from the data folder using LlamaIndex."""
    has_files = any(
        os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
        for _, _, files in os.walk(data_dir) for f in files
    )
    if not has_files:
        print(f"No supported files found in '{data_dir}'.")
        return []

    # Without this package PDFs/DOCX are read as raw bytes instead of text
    try:
        import llama_index.readers.file  # noqa: F401
    except ImportError:
        raise ImportError("PDF/DOCX reader missing. Run:  pip install llama-index-readers-file pypdf docx2txt")

    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,
        required_exts=SUPPORTED_EXTS,
    )
    return reader.load_data()


def load_web_documents(urls: List[str]) -> List[Document]:
    """Download web pages and convert them into LlamaIndex Documents."""
    web_docs = []
    for url in urls:
        try:
            response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # remove non-content tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = " ".join(soup.get_text(separator=" ").split())
            if text:
                web_docs.append(Document(text=text, metadata={"file_name": url, "source_type": "web"}))
                print(f"Loaded web page: {url}")
        except Exception as e:
            print(f"Could not load {url}: {e}")
    return web_docs


def load_all_documents() -> List[Document]:
    return load_local_documents(DATA_DIR) + load_web_documents(load_saved_urls())


# ======================================================================================
# Step 6: Build (or load) the vector index with LlamaIndex
# ======================================================================================
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage


def get_sources_fingerprint() -> str:
    """A hash that changes whenever files, URLs or index settings change."""
    items = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                path = os.path.join(root, f)
                stat = os.stat(path)
                items.append(f"{path}|{stat.st_size}|{int(stat.st_mtime)}")
    items += [f"url|{u}" for u in load_saved_urls()]
    items.append(f"settings|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{EMBED_MODEL_NAME}|v{INDEX_VERSION}")
    return hashlib.md5("\n".join(sorted(items)).encode()).hexdigest()


def build_or_load_index(force_rebuild: bool = False) -> VectorStoreIndex:
    """Load the saved index if documents did not change, otherwise rebuild it."""
    fingerprint = get_sources_fingerprint()

    saved_fingerprint = None
    if os.path.exists(FINGERPRINT_FILE):
        with open(FINGERPRINT_FILE) as f:
            saved_fingerprint = json.load(f).get("fingerprint")

    if not force_rebuild and saved_fingerprint == fingerprint:
        print("Loading existing index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)

    print("Building a new index...")
    docs = load_all_documents()
    if not docs:
        raise ValueError(f"No documents to index. Add files to '{DATA_DIR}' or add a web URL.")

    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(persist_dir=STORAGE_DIR)

    with open(FINGERPRINT_FILE, "w") as f:
        json.dump({"fingerprint": fingerprint}, f)

    return index


def list_indexed_documents(index: VectorStoreIndex) -> List[str]:
    """Names of all documents present in the index (used for multi-document features)."""
    return sorted({node.metadata.get("file_name", "unknown") for node in index.docstore.docs.values()})


# ======================================================================================
# Step 7: Semantic retrieval with the LlamaIndex Retriever
# ======================================================================================
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator, FilterCondition


def get_retriever(index: VectorStoreIndex, top_k: int = TOP_K, file_names: Optional[List[str]] = None):
    """LlamaIndex retriever. If file_names is given, search only inside those documents."""
    filters = None
    if file_names:
        # One "file_name == X" filter per document, combined with OR
        # (simple string comparisons work the same in every LlamaIndex version)
        filters = MetadataFilters(
            filters=[MetadataFilter(key="file_name", value=str(name), operator=FilterOperator.EQ)
                     for name in file_names],
            condition=FilterCondition.OR,
        )
    return index.as_retriever(similarity_top_k=top_k, filters=filters)


# ======================================================================================
# Step 8: Connect LlamaIndex to LangChain
# ======================================================================================
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def source_label(node) -> str:
    name = node.metadata.get("file_name", "unknown")
    page = node.metadata.get("page_label")
    return f"{name}, page {page}" if page else name


def format_nodes(nodes) -> str:
    """Convert retrieved LlamaIndex nodes into a context string for the prompt."""
    if not nodes:
        return "No relevant context found."
    blocks = []
    for i, node in enumerate(nodes, start=1):
        blocks.append(f"[Source {i}: {source_label(node)}]\n{node.get_content().strip()}")
    return "\n\n".join(blocks)


def make_context_runnable(retriever):
    """LlamaIndex retriever  ->  LangChain Runnable that returns the context string."""
    return RunnableLambda(lambda question: format_nodes(retriever.retrieve(question)))


# ======================================================================================
# Step 9: Prompt templates (LangChain)
# ======================================================================================
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are an intelligent Document Assistant.
Answer the user's question using ONLY the context provided below.

Rules:
1. Use only the information in the context. Never fabricate information.
2. If the context does not contain the answer, say exactly: "I could not find this information in the provided documents."
3. Cite the sources you used in square brackets, for example [Source 1] or [Source 2].
4. Keep the answer clear and well structured. Use bullet points where helpful."""),
    ("user",
     """Context:
{context}

Question: {question}

Answer:""")
])


# ======================================================================================
# Step 10: RAG chain using LCEL (LangChain Expression Language)
# ======================================================================================
from langchain_core.output_parsers import StrOutputParser


def build_rag_chain(retriever, llm):
    return (
        {
            "context": make_context_runnable(retriever),   # LlamaIndex retrieval
            "question": RunnablePassthrough(),             # user question as-is
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )


# ======================================================================================
# Step 11: Structured output with PydanticOutputParser
# ======================================================================================
from langchain_core.output_parsers import PydanticOutputParser


class RAGAnswer(BaseModel):
    answer: str = Field(description="The answer to the question, based only on the context")
    sources: List[str] = Field(description="Document names used for the answer, e.g. ['policy.pdf, page 2']")
    confidence: Literal["high", "medium", "low"] = Field(description="How well the context supports the answer")
    follow_up_questions: List[str] = Field(description="Two short follow-up questions the user might ask next")


pydantic_parser = PydanticOutputParser(pydantic_object=RAGAnswer)

structured_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are an intelligent Document Assistant. Answer using ONLY the context.
If the context does not contain the answer, say so in the answer field and set confidence to "low".

{format_instructions}"""),
    ("user", "Context:\n{context}\n\nQuestion: {question}")
]).partial(format_instructions=pydantic_parser.get_format_instructions())


def generate_structured_answer(context: str, question: str, llm, fallback_sources: List[str]) -> RAGAnswer:
    """Structured answer. Falls back to plain text if the LLM output can't be parsed."""
    inputs = {"context": context, "question": question}
    try:
        return invoke_with_retry(structured_prompt | llm | pydantic_parser, inputs)
    except Exception as e:
        if is_rate_limit_error(e):
            raise
        text = invoke_with_retry(qa_prompt | llm | StrOutputParser(), inputs)
        return RAGAnswer(answer=text, sources=fallback_sources, confidence="medium", follow_up_questions=[])


def ask_structured(question: str, retriever, llm) -> RAGAnswer:
    nodes = retriever.retrieve(question)
    return generate_structured_answer(format_nodes(nodes), question, llm,
                                      sorted({source_label(n) for n in nodes}))


# ======================================================================================
# Step 12: Multi-document querying (compare documents)
# ======================================================================================
compare_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a Document Analyst. You receive context grouped by document.
Answer the question for EACH document separately and then give a short comparison
(similarities and differences). Use ONLY the given context. If a document does not cover
the topic, say "Not covered in this document". Use a markdown table where helpful."""),
    ("user", "{context}\n\nQuestion: {question}")
])


def compare_documents(index, question: str, file_names: List[str], llm, top_k: int = 3) -> str:
    sections = []
    for name in file_names:
        nodes = get_retriever(index, top_k=top_k, file_names=[name]).retrieve(question)
        sections.append(f"=== DOCUMENT: {name} ===\n{format_nodes(nodes)}")
    compare_chain = compare_prompt | llm | StrOutputParser()
    return invoke_with_retry(compare_chain, {"context": "\n\n".join(sections), "question": question})


# ======================================================================================
# Step 13: Document summarization (Map-Reduce with LangChain)
# ======================================================================================
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at summarizing documents accurately. Use only the given text."),
    ("user",
     """Summarize the following text from the document "{doc_name}".

Give:
1. A one-paragraph overview
2. Key points as bullet points (include important numbers, limits and deadlines)

Text:
{text}""")
])

combine_prompt = ChatPromptTemplate.from_messages([
    ("system", "You combine partial summaries into one clear final summary without repeating points."),
    ("user",
     """These are partial summaries of the document "{doc_name}":

{text}

Write the final summary with:
1. A one-paragraph overview
2. Key points as bullet points""")
])


def get_document_text_chunks(index: VectorStoreIndex, file_name: str) -> List[str]:
    return [node.get_content() for node in index.docstore.docs.values()
            if node.metadata.get("file_name") == file_name]


def group_texts(texts: List[str], max_chars: int = MAX_CHARS_PER_CALL) -> List[str]:
    """Join texts into groups of at most max_chars characters (long single texts are split)."""
    groups, current = [], ""
    for text in texts:
        while len(text) > max_chars:                 # split a very long text
            if current:
                groups.append(current)
                current = ""
            groups.append(text[:max_chars])
            text = text[max_chars:]
        if current and len(current) + len(text) > max_chars:
            groups.append(current)
            current = ""
        current += text + "\n\n"
    if current.strip():
        groups.append(current)
    return groups


def summarize_document(index, file_name: str, llm, progress_callback=None) -> str:
    chunks = get_document_text_chunks(index, file_name)
    if not chunks:
        return f"Document '{file_name}' not found in the index."

    summary_chain = summary_prompt | llm | StrOutputParser()
    combine_chain = combine_prompt | llm | StrOutputParser()

    groups = group_texts(chunks)

    # Short document -> single call
    if len(groups) == 1:
        return invoke_with_retry(summary_chain, {"doc_name": file_name, "text": groups[0]})

    # Long document -> MAP: summarize each group one by one (stays under the rate limit)
    partial_summaries = []
    for i, group in enumerate(groups, start=1):
        message = f"Summarizing part {i} of {len(groups)}..."
        print(message)
        if progress_callback:
            progress_callback(i / len(groups), message)
        partial_summaries.append(invoke_with_retry(summary_chain, {"doc_name": file_name, "text": group}))

    # REDUCE: combine partial summaries (repeat while they are still too long for one call)
    while len(partial_summaries) > 1:
        combined_groups = group_texts(partial_summaries)
        if len(combined_groups) == 1:
            break
        partial_summaries = [invoke_with_retry(combine_chain, {"doc_name": file_name, "text": g})
                             for g in combined_groups]

    return invoke_with_retry(combine_chain, {"doc_name": file_name,
                                             "text": "\n\n---\n\n".join(partial_summaries)})


# ======================================================================================
# Step 14: Conversational Document Assistant (with chat history)
# ======================================================================================
from langchain_core.messages import HumanMessage, AIMessage

condense_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Given the chat history and a follow-up question, rewrite the follow-up question
into a standalone question that can be understood without the chat history.
Do NOT answer the question. Return ONLY the rewritten question."""),
    MessagesPlaceholder("chat_history"),
    ("user", "{question}")
])


SUMMARY_WORDS = ["summary", "summarize", "summarise", "overview", "summarization"]


def is_summary_request(question: str) -> bool:
    """Chat retrieval only sees the top-k chunks, so 'summarize the document' needs summarize_document()."""
    return any(word in question.lower() for word in SUMMARY_WORDS)


def ask(question: str, chat_history: list, retriever, llm, structured: bool = False) -> dict:
    """Conversational RAG: condense question -> retrieve (LlamaIndex) -> answer (LangChain)."""
    standalone_question = question
    if chat_history:
        condense_chain = condense_prompt | llm | StrOutputParser()
        standalone_question = invoke_with_retry(condense_chain, {"chat_history": chat_history, "question": question})

    nodes = retriever.retrieve(standalone_question)
    context = format_nodes(nodes)
    sources = [{"source": source_label(n), "score": round(n.score or 0, 3),
                "text": n.get_content()[:400]} for n in nodes]

    result = {"question": question, "standalone_question": standalone_question,
              "sources": sources, "confidence": None, "follow_up_questions": []}

    if structured:
        parsed = generate_structured_answer(context, standalone_question, llm,
                                            sorted({s["source"] for s in sources}))
        result.update(answer=parsed.answer, confidence=parsed.confidence,
                      follow_up_questions=parsed.follow_up_questions)
    else:
        answer_chain = qa_prompt | llm | StrOutputParser()
        result["answer"] = invoke_with_retry(answer_chain, {"context": context, "question": standalone_question})

    return result


# ======================================================================================
# Step 15: Streamlit Web App (UI)
# ======================================================================================
def friendly_error(error: Exception) -> str:
    if is_rate_limit_error(error):
        return ("The LLM provider's rate limit was reached (Groq free tier allows only a limited number of "
                "tokens per minute). Please wait a minute and try again, or reduce 'Chunks to retrieve (top-k)'.")
    return f"Something went wrong: {error}"


def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")
    st.header("📄 RAG-Based Intelligent Document Assistant")
    st.caption("LlamaIndex for indexing & retrieval · LangChain for LLM orchestration, prompts & output parsing")

    if missing_api_key():
        st.error(f"{missing_api_key()} is missing. Add it to your .env file and restart the app.")
        st.stop()

    # ---------- Cached resources (loaded once, not on every click) ----------
    @st.cache_resource(show_spinner="Loading embedding model...")
    def setup_llamaindex():
        configure_llamaindex()
        return True

    @st.cache_resource(show_spinner="Indexing documents with LlamaIndex...")
    def get_index(fingerprint: str):
        # 'fingerprint' is the cache key: when documents change, the index is rebuilt
        return build_or_load_index()

    @st.cache_resource
    def get_cached_llm():
        return get_llm()

    setup_llamaindex()
    llm = get_cached_llm()

    # ---------- Sidebar: knowledge base ----------
    with st.sidebar:
        st.subheader("📚 Knowledge base")

        uploaded_files = st.file_uploader("Upload documents", type=[e[1:] for e in SUPPORTED_EXTS],
                                          accept_multiple_files=True)
        if uploaded_files and st.button("Add uploaded files", use_container_width=True):
            for up in uploaded_files:
                with open(os.path.join(DATA_DIR, up.name), "wb") as f:
                    f.write(up.getbuffer())
            st.success(f"Added {len(uploaded_files)} file(s). Re-indexing...")
            st.rerun()

        new_url = st.text_input("Add a web page URL", placeholder="https://...")
        if st.button("Add URL", use_container_width=True) and new_url.strip():
            add_url(new_url)
            st.rerun()

        if load_saved_urls() and st.button("Remove all URLs", use_container_width=True):
            save_urls([])
            st.rerun()

        top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, TOP_K)
        structured_mode = st.toggle("Structured answers (Pydantic parser)", value=False)

        if st.button("🔄 Force rebuild index", use_container_width=True):
            if os.path.exists(FINGERPRINT_FILE):
                os.remove(FINGERPRINT_FILE)
            get_index.clear()
            st.rerun()

    # ---------- Build / load index ----------
    try:
        index = get_index(get_sources_fingerprint())
    except ValueError as e:
        st.info(str(e))
        st.stop()

    available_docs = list_indexed_documents(index)

    with st.sidebar:
        st.markdown(f"**Indexed documents ({len(available_docs)})** · {len(index.docstore.docs)} chunks")
        for name in available_docs:
            st.markdown(f"- {name}")

    tab_chat, tab_compare, tab_summary = st.tabs(["💬 Ask questions", "⚖️ Compare documents", "📝 Summarize"])

    # ---------- Tab 1: Chat ----------
    with tab_chat:
        selected = st.multiselect("Search in (leave empty = all documents)", available_docs)

        if "messages" not in st.session_state:
            st.session_state.messages = []     # for display
            st.session_state.history = []      # for LangChain (HumanMessage / AIMessage)

        if st.session_state.messages and st.button("Clear chat"):
            st.session_state.messages, st.session_state.history = [], []
            st.rerun()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources"):
                        for s in msg["sources"]:
                            st.markdown(f"**{s['source']}** (score {s['score']})\n\n> {s['text']}...")

        question = st.chat_input("Ask a question about your documents")
        if question:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                # Whole-document summary requests -> summarize_document() instead of top-k retrieval
                summary_target = None
                if is_summary_request(question):
                    if len(selected) == 1:
                        summary_target = selected[0]
                    elif len(available_docs) == 1:
                        summary_target = available_docs[0]

                if is_summary_request(question) and summary_target is None:
                    response = {"answer": "To summarize a whole document, select **one** document in "
                                          "'Search in' above (or use the **Summarize** tab) and ask again.",
                                "sources": [], "confidence": None, "follow_up_questions": [],
                                "standalone_question": question}
                elif summary_target:
                    with st.spinner(f"Summarizing {summary_target}..."):
                        try:
                            summary = summarize_document(index, summary_target, llm)
                        except Exception as e:
                            st.error(friendly_error(e))
                            st.stop()
                    response = {"answer": f"**Summary of {summary_target}**\n\n{summary}", "sources": [],
                                "confidence": None, "follow_up_questions": [], "standalone_question": question}
                else:
                    with st.spinner("Retrieving and generating..."):
                        retriever = get_retriever(index, top_k=top_k, file_names=selected or None)
                        try:
                            response = ask(question, st.session_state.history, retriever, llm, structured_mode)
                        except Exception as e:
                            st.error(friendly_error(e))
                            st.stop()

                content = response["answer"]
                if response["confidence"]:
                    content += f"\n\n**Confidence:** {response['confidence']}"
                if response["follow_up_questions"]:
                    content += "\n\n**You could also ask:**\n" + "\n".join(
                        f"- {q}" for q in response["follow_up_questions"])
                if response["standalone_question"] != question:
                    content += f"\n\n_Interpreted as: {response['standalone_question']}_"

                st.markdown(content)
                if response["sources"]:
                    with st.expander("Sources"):
                        for s in response["sources"]:
                            st.markdown(f"**{s['source']}** (score {s['score']})\n\n> {s['text']}...")

            st.session_state.messages += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": content, "sources": response["sources"]},
            ]
            st.session_state.history += [HumanMessage(content=question),
                                         AIMessage(content=response["answer"])]

    # ---------- Tab 2: Compare ----------
    with tab_compare:
        docs_to_compare = st.multiselect("Documents to compare", available_docs,
                                         default=available_docs[:2] if len(available_docs) >= 2 else available_docs)
        compare_q = st.text_input("Comparison question", placeholder="e.g. What are the approval rules and limits?")
        if st.button("Compare", type="primary"):
            if len(docs_to_compare) < 2 or not compare_q.strip():
                st.error("Select at least two documents and enter a question.")
            else:
                with st.spinner("Comparing documents..."):
                    try:
                        st.markdown(compare_documents(index, compare_q, docs_to_compare, llm))
                    except Exception as e:
                        st.error(friendly_error(e))

    # ---------- Tab 3: Summarize ----------
    with tab_summary:
        doc_to_summarize = st.selectbox("Document", available_docs)
        st.caption("Long documents are summarized part by part. On the Groq free tier this can take a few minutes.")
        if st.button("Summarize", type="primary"):
            progress = st.progress(0.0, text="Summarizing...")
            try:
                with st.spinner("Summarizing..."):
                    summary = summarize_document(index, doc_to_summarize, llm,
                                                 progress_callback=lambda p, msg: progress.progress(p, text=msg))
                progress.empty()
                st.markdown(summary)
            except Exception as e:
                progress.empty()
                st.error(friendly_error(e))


# Run the web app only when started with:  streamlit run rag_document_assistant.py
if __name__ == "__main__":
    from streamlit import runtime
    if runtime.exists():
        run_streamlit_app()
    else:
        print("This is a Streamlit app. Start it with:\n\n    streamlit run rag_document_assistant.py\n")
