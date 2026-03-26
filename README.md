# Autonomous Project Manager (APM)

A demo-grade AI system that ingests construction project files, cross-checks them against the Dubai Building Code, and produces a grounded compliance risk report — all running locally with no cloud API keys.

---

## Architecture Overview

```text
User prompt (LangSmith Studio or CLI)
        │
        ▼
┌─────────────────────────────────────────────┐
│             APM ReAct Agent                 │
│          (LangGraph + LangChain)            │
│                                             │
│  Tool 1: read_project_files                 │
│    └─ loads meeting minutes, SOP, logs      │
│                                             │
│  Tool 2: retrieve_building_code (RAG)       │
│    └─ semantic search over Dubai Code PDF   │
│         via ChromaDB + local embeddings     │
│                                             │
│  LLM: qwen3:0.6b via Ollama (local)         │
│    └─ reasoning=True (thinking tokens)      │
└─────────────────────────────────────────────┘
        │
        ▼
Markdown compliance report with [1][2] citations
```

---

## Design Choices

### 1. Local-first, no API keys

All inference runs on-device via [Ollama](https://ollama.com). The chat model (`qwen3:0.6b`) and embedding model (`qwen3-embedding:0.6b`) are pulled locally. This avoids cloud costs and keeps project data private.

### 2. RAG over the Building Code

The Dubai Building Code PDF (843 pages) is chunked, embedded, and stored in a persistent [ChromaDB](https://www.trychroma.com/) collection on first run. Subsequent runs skip re-indexing. The agent queries this store with targeted natural-language queries (FAR limits, parking ratios, setback rules, energy codes) and receives page-cited excerpts.

### 3. ReAct agent via LangGraph

The agent is built with `langchain.agents.create_agent`, which produces a compiled `StateGraph`. The ReAct loop (reason → act → observe → reason…) means the agent decides when to call tools and how many times. LangGraph Studio exposes each step as a visible graph node, making the trace inspectable during the demo.

### 4. Two-tool design

- **`read_project_files`** — deterministically loads the three project files (meeting minutes, firm SOP, system logs) and returns their raw text. No parsing guesswork; the LLM reads them.
- **`retrieve_building_code`** — semantic search returning the top-4 most relevant code passages with page numbers, used as grounding evidence.

### Tool Function I/O Contracts

These contracts are intentionally reflected in the tool docstrings because the
LLM uses those docstrings to decide when and how to call tools.

#### `read_project_files`

- Inputs: none
- Output type: `str`
- Output shape:
        - A single string with labeled sections for:
                - meeting notes
                - internal policy thresholds
                - system logs
- Expected usage:
        - Call once at the beginning of each run to collect project context.

#### `retrieve_building_code`

- Inputs:
        - `query: str`
- Output type:
        - `tuple[str, list]` (`response_format="content_and_artifact"`)
- Output shape:
        - `content` (`str`): concatenated code snippets in citation-friendly form
        - `artifact` (`list`): retrieved document objects (metadata + content)
- Expected usage:
        - Call multiple times with targeted compliance questions (FAR, parking,
                setback, energy) before writing conclusions.

### 5. Forced citation discipline in the prompt

The system prompt instructs the agent to make at least 3 `retrieve_building_code` calls before writing the report. The final output uses `[1]` `[2]` inline citations with a `## References` section — keeping the small model honest about evidence provenance.

### 6. Reasoning tokens enabled

`reasoning=True` on the Ollama chat model exposes the model's thinking as a separate `reasoning_content` field in the response, visible in LangSmith traces. This lets reviewers see the model's internal reasoning without it bleeding into the final answer.

---

## Project Structure

```text
.
├── data/
│   ├── Dubai_Zoning_Codes.pdf      # Building code (indexed into ChromaDB)
│   ├── meeting_minutes_alpha.txt   # Project meeting minutes
│   ├── firm_sop_v3.json            # Firm SOP thresholds
│   └── system_logs.log             # System/CAD logs
├── src/
│   └── apm/
│       └── agent.py                # Agent, tools, system prompt, graph factory
├── chroma/                         # Persisted ChromaDB vector store (auto-created)
├── main.py                         # CLI entrypoint
├── langgraph.json                  # LangGraph Studio config
└── pyproject.toml                  # Dependencies (managed by uv)
```

---

## Prerequisites

| Requirement                      | Version    | Notes                         |
| -------------------------------- | ---------- | ----------------------------- |
| Python                           | 3.12+      | Managed via `.python-version` |
| [uv](https://docs.astral.sh/uv/) | any recent | Fast Python package manager   |
| [Ollama](https://ollama.com)     | 0.18+      | Local LLM inference server    |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/nithinjp1997/autonomous-project-managament.git
cd autonomous-project-managament
uv sync
```

### 2. Install and start Ollama

```bash
# Install (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b
ollama serve
```

Ollama runs as a background service on `http://localhost:11434` after install.

### 3. Add your data files

Place these files in the `data/` directory:

```text
data/Dubai_Zoning_Codes.pdf
data/meeting_minutes_alpha.txt
data/firm_sop_v3.json
data/system_logs.log
```

> On first run the agent will index the PDF into ChromaDB.

---

## Running the Agent

### Option A — LangSmith Studio (recommended for demo)

LangSmith Studio provides a visual graph trace showing each tool call and reasoning step.

```bash
uv run langgraph dev
```

Then open [Studio UI](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024) in your browser.

In the Studio chat, type:

```text
Analyse this project for compliance risks and produce a report.
```

The agent will:

1. Call `read_project_files`
2. Call `retrieve_building_code` 3–4 times with targeted queries
3. Return a markdown report with cited evidence

### Option B — CLI

```bash
uv run python main.py
```

Outputs the compliance report directly to the terminal.

---

## Notes

- The ChromaDB collection is persisted in `chroma/`. Delete this folder to force a full re-index of the PDF.
- `qwen3:0.6b` is small and fast but may occasionally skip tool calls; restarting the conversation usually resolves this.
- No LangSmith API key is required to use Studio locally in dev mode.
