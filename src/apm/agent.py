"""Autonomous Project Manager - single ReAct agent with retrieval grounding."""

import json
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CHROMA_DIR = Path(__file__).resolve().parents[2] / "chroma"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
CHAT_MODEL = "qwen3:0.6b"

# ---------------------------------------------------------------------------
# Vector store (singleton - reused across invocations)
# ---------------------------------------------------------------------------
_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

_vector_store = Chroma(
    collection_name="zoning_codes",
    embedding_function=_embeddings,
    persist_directory=str(CHROMA_DIR),
)


def _ensure_indexed() -> None:
    """Index the PDF into Chroma if the collection is empty."""
    if _vector_store._collection.count() > 0:
        return
    loader = PyPDFLoader(str(DATA_DIR / "Dubai_Zoning_Codes.pdf"))
    pages = loader.load_and_split()
    _vector_store.add_documents(pages)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def read_project_files() -> str:
    """Load all internal project documents into a single text payload.

    Call this tool before compliance analysis so findings are grounded
    in actual project data rather than assumptions.

    Returns:
        str: A single string containing three labeled sections —
            meeting notes, internal policy thresholds, and system logs.
    """
    files = {
        "meeting_minutes": DATA_DIR / "meeting_minutes_alpha.txt",
        "firm_sop": DATA_DIR / "firm_sop_v3.json",
        "system_logs": DATA_DIR / "system_logs.log",
    }
    sections = []
    for label, path in files.items():
        content = path.read_text()
        if path.suffix == ".json":
            content = json.dumps(json.loads(content), indent=2)
        sections.append(f"=== {label} ({path.name}) ===\n{content}")
    return "\n\n".join(sections)


@tool(response_format="content_and_artifact")
def retrieve_building_code(query: str) -> tuple[str, list]:
    """Retrieve relevant Dubai Building Code excerpts for a focused query.

    Prefer narrow, targeted queries (e.g. FAR limits, parking requirements,
    setback rules, energy criteria) so returned citations are precise.

    Args:
        query: A specific regulatory question to search against the
            Dubai Building Code PDF.

    Returns:
        tuple[str, list]: A two-element tuple where:
            - content (str): Readable snippets with source page labels.
            - artifact (list): Retrieved Document objects for traceability.
    """
    retrieved_docs = _vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        f"Source: page {doc.metadata.get('page', '?')} | {doc.metadata.get('source', '')}\n"
        f"Content: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an Autonomous Project Manager (APM) agent responsible for ensuring that a construction project complies with the Dubai Building Code. Answer the user's queries by using the tools at your disposal. Your task is to analyze and flag any potential compliance issues. Generate a well drafted response to the user queries.

When your answer references information from the building code or project files, add inline citations using [1], [2], etc. At the end of your response, list all citations under a "References" heading with the source name and page number (if applicable).
"""


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def make_graph():
    """Build and return the APM agent graph."""
    _ensure_indexed()

    llm = init_chat_model(
        f"ollama:{CHAT_MODEL}",
        reasoning=True,
    )

    graph = create_agent(
        model=llm,
        tools=[read_project_files, retrieve_building_code],
        system_prompt=SYSTEM_PROMPT,
    )
    return graph
