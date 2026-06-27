# Nexus

A self-hosted, local-first AI agent workspace built from scratch — tool-using agents, advanced retrieval-augmented generation (RAG), persistent memory, web search, full observability, and a streaming chat UI with conversation history. Runs on OpenAI or fully offline on local models. Inspired by the local-first philosophy of self-hosted AI workspaces.

**Version 1.1**

---

## What it does

Nexus is an autonomous AI agent that can:

- **Reason and act in a loop** — a LangGraph ReAct agent that decides when to call tools, reads the results, and continues until it can answer.
- **Operate on your system** — read files, write files, list directories, run shell commands.
- **Search the live web** — answer questions about current events and anything beyond its training data (DuckDuckGo, no API key).
- **Answer from your own documents** — advanced RAG: hybrid search (semantic + keyword), Reciprocal Rank Fusion, cross-encoder reranking, and LLM query expansion, with citations.
- **Remember you** — short-term conversation memory (per thread) and long-term memory (durable facts that persist across every session).
- **Run locally or in the cloud** — built on an OpenAI-compatible provider layer; switch between OpenAI and a local model (Ollama) with a single config change, no code edits.
- **Monitor itself** — every run records latency, token usage, cost, and success/failure to Postgres, surfaced in a live Grafana dashboard.
- **Chat like a real app** — streaming web UI with a conversation sidebar; past chats persist and reopen.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Web UI (streaming chat + conversation sidebar)           │
│     FastAPI backend  ·  SSE token streaming  ·  HTML/JS   │
├──────────────────────────────────────────────────────────┤
│  Agent core (LangGraph ReAct loop)                        │
│     ├─ LLM provider layer (OpenAI ⇄ Ollama)               │
│     ├─ Tools: file I/O · shell · web search               │
│     │         · document search (RAG) · memory            │
│     └─ Prompt + tool-selection management                 │
├──────────────────────────────────────────────────────────┤
│  Advanced RAG                                             │
│     ingest → chunk → embed → pgvector                     │
│     query → expand → hybrid (dense+sparse) → RRF          │
│             → cross-encoder rerank → top-k → cite         │
├──────────────────────────────────────────────────────────┤
│  Memory: short-term (SQLite checkpointer)                 │
│          long-term (Postgres key-value)                   │
├──────────────────────────────────────────────────────────┤
│  Observability: per-run metrics → Postgres → Grafana      │
├──────────────────────────────────────────────────────────┤
│  Infra: Docker Compose (Postgres/pgvector + Grafana)      │
└──────────────────────────────────────────────────────────┘
```

---

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.11 | Standard for the AI/LLM ecosystem |
| Agent framework | LangChain + LangGraph | Stateful tool-calling agent loop with checkpointing |
| LLM (cloud) | OpenAI (`gpt-4o-mini`) | Strong, cost-effective default |
| LLM (local) | Ollama (`qwen3` family) | Offline, zero-cost, private; reliable tool-calling |
| Provider layer | OpenAI-compatible | One abstraction, cloud or local via `.env` |
| Vector store | Postgres + pgvector | Vectors, full-text, metadata, and app state in one DB |
| Embeddings | OpenAI `text-embedding-3-small` | Cost-effective retrieval quality |
| Retrieval | Hybrid (pgvector + Postgres FTS) + RRF + cross-encoder rerank | Production-grade RAG pipeline |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local, fast, precise re-scoring |
| Web search | DuckDuckGo (`ddgs`) | Free, no API key |
| Short-term memory | LangGraph SQLite checkpointer | Per-conversation state |
| Long-term memory | Postgres key-value table | Durable cross-session facts |
| Web backend | FastAPI + SSE | Async streaming chat endpoint |
| Tracing (dev) | LangSmith | Full execution traces |
| Metrics (prod) | Custom recorder → Postgres → Grafana | Local-first observability |
| Infrastructure | Docker Compose | Postgres + Grafana with one command |

---

## Project structure

```
nexus/
├── app/
│   ├── core/
│   │   ├── config.py            # typed settings, loaded once from .env
│   │   ├── llm.py               # swappable LLM provider (OpenAI/Ollama)
│   │   ├── pricing.py           # token → cost conversion
│   │   └── metrics.py           # per-run metrics recorder
│   ├── tools/
│   │   ├── os_tools.py          # read/write file, list dir, run shell
│   │   ├── web_search.py        # DuckDuckGo web search
│   │   ├── rag_tool.py          # document search (advanced RAG pipeline)
│   │   └── memory_tools.py      # save/load long-term memory
│   ├── agent/
│   │   ├── state.py             # graph state (message history)
│   │   ├── graph.py             # LangGraph agent loop, tool binding
│   │   └── agent_demo.py        # run_agent() entry point
│   ├── rag/
│   │   ├── loader.py            # documents → text
│   │   ├── chunker.py           # recursive chunking
│   │   ├── semantic_chunker.py  # semantic (meaning-based) chunking
│   │   ├── store.py             # embed + store in pgvector
│   │   ├── retriever.py         # dense similarity search
│   │   ├── hybrid.py            # hybrid search + RRF + rerank + expansion
│   │   ├── rerank.py            # cross-encoder reranking
│   │   └── query_rewrite.py     # LLM query expansion
│   ├── memory/
│   │   └── long_term.py         # Postgres-backed durable memory
│   └── web/
│       ├── server.py            # FastAPI: chat (streaming) + conversation API
│       ├── conversations.py     # conversation/message persistence
│       └── index.html           # chat UI with sidebar
├── docker/
│   ├── docker-compose.yml       # Postgres/pgvector + Grafana
│   └── grafana/provisioning/    # auto-provisioned data source
├── Modelfile.qwen               # Ollama model variant (extended context)
├── .env.example                 # configuration template
└── README.md
```

---

## Quick start

### Prerequisites
- Python 3.11+
- Docker (Docker Desktop + WSL integration on Windows)
- An OpenAI API key (optional if running fully local with Ollama)

### 1. Environment
```bash
git clone https://github.com/<your-username>/Nexus.git
cd Nexus
conda create -n nexus python=3.11 -y
conda activate nexus
pip install langchain langchain-openai langgraph langchain-core \
            langgraph-checkpoint-sqlite langchain-postgres langchain-community \
            langchain-experimental "psycopg[binary]" pypdf pydantic \
            pydantic-settings python-dotenv tiktoken openai \
            ddgs sentence-transformers fastapi "uvicorn[standard]" sse-starlette
```

### 2. Configure
```bash
cp .env.example .env   # add OPENAI_API_KEY (or configure Ollama — see below)
```

### 3. Infrastructure
```bash
docker compose -f docker/docker-compose.yml up -d
# Postgres (pgvector) on localhost:5434
# Grafana on http://localhost:3001  (admin / admin)

# enable pgvector + full-text index (one-time)
docker exec claude_desktop_pg psql -U claude -d claude_desktop -c "
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE IF EXISTS langchain_pg_embedding
  ADD COLUMN IF NOT EXISTS fts tsvector
  GENERATED ALWAYS AS (to_tsvector('english', document)) STORED;
CREATE INDEX IF NOT EXISTS idx_fts ON langchain_pg_embedding USING GIN (fts);
"
```

### 4. Run
```bash
# ingest a document into the vector store
python -m app.rag.ingest_demo

# launch the chat UI
python -m app.web.server
# open http://localhost:8000
```

---

## Running fully local (Ollama)

Nexus uses an OpenAI-compatible provider layer, so switching to a local model is a config change only.

```bash
# install Ollama, pull a tool-capable model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b

# (recommended) build a variant with a larger context window for agent loops
ollama create qwen3-agent -f Modelfile.qwen
```

Then in `.env`:
```
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3-agent
```
No code changes. Switch back to OpenAI by setting `LLM_PROVIDER=openai`, blanking `LLM_BASE_URL`, and `LLM_MODEL=gpt-4o-mini`.

---

## Advanced RAG pipeline

Retrieval runs a full modern pipeline:

1. **Query expansion** — the LLM rewrites the question into multiple focused search queries.
2. **Hybrid search** — each query runs dense (pgvector semantic) and sparse (Postgres full-text) search.
3. **Reciprocal Rank Fusion** — the two ranked lists are fused; agreement floats to the top.
4. **Cross-encoder reranking** — candidates are re-scored against the original question for precision.
5. **Citation** — top chunks are returned numbered and sourced for the LLM to cite.

Chunking supports both recursive (fast) and semantic (meaning-based) strategies.

---

## Monitoring

Every agent run records metrics (duration, tokens, cost, success) to the `run_metrics` table. The Grafana dashboard at `http://localhost:3001` shows cost over time, total spend, run count, p95 latency, error rate, and token usage. The Postgres data source is auto-provisioned.

---

## Roadmap

- [ ] MCP (Model Context Protocol) integration — connect to the ecosystem of external tool servers
- [ ] Skills system — packaged, progressively-disclosed domain capabilities
- [ ] Eval harness — automated regression testing of agent behavior
- [ ] Containerize the app itself (full one-command stack)
- [ ] Model routing (local for simple tasks, cloud for hard ones)
- [ ] Electron/Tauri desktop packaging

---

## License

MIT