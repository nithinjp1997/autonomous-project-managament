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
    """Read all project data files (meeting minutes, SOP, system logs).

    Call this tool FIRST to load the project context before doing any analysis.
    Returns the contents of meeting minutes, firm SOP, and system logs.
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
def retrieve_building_code(query: str):
    """Search the Dubai Building Code for regulations, limits, or standards.

    Use this tool whenever you need to look up zoning rules, FAR limits,
    setback requirements, parking ratios, energy codes, or any other
    regulatory constraint from the official Dubai Building Code PDF.
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
SYSTEM_PROMPT = """\
You are APM, a construction compliance analyst.

You have 2 tools:
- read_project_files: returns internal project documents.
- retrieve_building_code: returns Dubai code excerpts with page numbers.

Follow this exact sequence:
1) Call read_project_files once.
2) Call retrieve_building_code at least 3 times with focused queries:
   - FAR / density limit
   - parking requirement
   - setback or balcony rule
   - energy efficiency rule (if relevant)
3) Write final answer in markdown.

Output format:
## Compliance Status
One line: compliant / non_compliant / needs_review

## Key Findings
- Finding with severity (HIGH/MEDIUM/LOW) and inline citation [1]
- Finding with inline citation [2]

## Data Conflicts
- Field mismatch: source A vs source B [3]

## Recommended Actions
- Action 1
- Action 2

## References
[1] Building code p.X - short quote
[2] Internal document - short quote
[3] Internal document - short quote
[4] Internal policy threshold - short quote

Rules:
- Do not answer until tools are called.
- Do not invent values, pages, quotes, or source names.
- Use source labels exactly as returned by tools.
- If evidence is missing, write "Insufficient evidence".
- Keep answer under 220 words.
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
