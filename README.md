# Nexus

A self-hosted, local-first AI agent workspace built from scratch — tool-using agents, advanced retrieval-augmented generation (RAG), persistent memory, web search, **Model Context Protocol (MCP) integration**, full observability, and a streaming chat UI with conversation history. Runs on OpenAI or fully offline on local models.

**Version 1.2**

---

## What it does

Nexus is an autonomous AI agent that can:

- **Reason and act in a loop** — a LangGraph ReAct agent that decides when to call tools, reads the results, and continues until it can answer.
- **Connect to the MCP ecosystem** — integrates external Model Context Protocol servers (filesystem, web fetch, MemPalace memory) alongside its own tools, all through one async agent loop.
- **Operate on your system** — run shell commands, and read/write/search files via the MCP filesystem server.
- **Search and fetch the web** — `web_search` (DuckDuckGo, no API key) to find pages; the MCP `fetch` server to read a specific URL.
- **Answer from your own documents** — advanced RAG: hybrid search (semantic + keyword), Reciprocal Rank Fusion, cross-encoder reranking, and LLM query expansion, with citations.
- **Remember you** — short-term conversation memory (per thread), long-term memory (Postgres), plus an optional local knowledge-graph memory via the MemPalace MCP server.
- **Run locally or in the cloud** — OpenAI-compatible provider layer; switch OpenAI ⇄ Ollama with one config change.
- **Monitor itself** — every run records latency, tokens, cost, and success to Postgres, surfaced in Grafana.
- **Chat like a real app** — streaming web UI with a conversation sidebar; past chats persist and reopen.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Web UI (async streaming chat + conversation sidebar)     │
│     FastAPI · SSE token streaming · async throughout      │
├──────────────────────────────────────────────────────────┤
│  Agent core (async LangGraph ReAct loop)                  │
│     ├─ LLM provider layer (OpenAI ⇄ Ollama)               │
│     ├─ Built-in tools: shell · web search                 │
│     │                  · document search (RAG) · memory   │
│     └─ MCP tools (loaded at startup, config-driven)       │
├──────────────────────────────────────────────────────────┤
│  MCP servers (stdio subprocesses)                         │
│     ├─ fetch        — read web pages                      │
│     ├─ filesystem   — file read/write/search/edit         │
│     └─ mempalace    — local knowledge-graph memory        │
├──────────────────────────────────────────────────────────┤
│  Advanced RAG: expand → hybrid (dense+sparse) → RRF       │
│                → cross-encoder rerank → cite              │
├──────────────────────────────────────────────────────────┤
│  Memory: short-term (async SQLite checkpointer)           │
│          long-term (Postgres) · MemPalace (MCP, optional) │
├──────────────────────────────────────────────────────────┤
│  Observability: per-run metrics → Postgres → Grafana      │
├──────────────────────────────────────────────────────────┤
│  Infra: Docker Compose (Postgres/pgvector + Grafana)      │
└──────────────────────────────────────────────────────────┘
```

---

## MCP integration

Nexus connects to external [Model Context Protocol](https://modelcontextprotocol.io) servers and exposes their tools to the agent alongside its own. Servers are declared in `mcp_servers.json` at the project root — adding one is a config edit, not a code change:

```json
{
  "fetch": {
    "transport": "stdio",
    "command": "uvx",
    "args": ["mcp-server-fetch"]
  },
  "filesystem": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
  },
  "mempalace": {
    "transport": "stdio",
    "command": "mempalace-mcp",
    "args": ["--palace", "/path/to/palace"]
  }
}
```

Tools are loaded at startup via `langchain-mcp-adapters`, which converts MCP tools into LangChain tools the agent loop uses natively. Loading is **resilient** — each server is loaded independently, so one failing server doesn't disable the others. Integrating MCP required making the entire request pipeline async (graph construction, streaming, and the SQLite checkpointer all use their async variants).

Requirements for MCP: Node.js 20+ (for `npx`-based servers), [`uv`](https://github.com/astral-sh/uv) (for `uvx`-based servers).

---

## Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.11 |
| Agent framework | LangChain + LangGraph (async) |
| MCP | langchain-mcp-adapters (fetch, filesystem, MemPalace servers) |
| LLM (cloud / local) | OpenAI `gpt-4o-mini` / Ollama `qwen3` |
| Vector store | Postgres + pgvector |
| Retrieval | Hybrid (pgvector + Postgres FTS) + RRF + cross-encoder rerank + query expansion |
| Web search | DuckDuckGo (`ddgs`) |
| Memory | LangGraph async SQLite checkpointer (short) · Postgres (long) · MemPalace (MCP, optional) |
| Web backend | FastAPI + SSE (async) |
| Observability | Custom metrics → Postgres → Grafana |
| Infrastructure | Docker Compose |

---

## Quick start

### Prerequisites
- Python 3.11+, Docker
- Node.js 20+ and `uv` (for MCP servers)
- OpenAI API key (optional if running fully local with Ollama)

### Setup
```bash
git clone https://github.com/vigoroth/Nexus.git
cd Nexus
conda create -n nexus python=3.11 -y
conda activate nexus
pip install langchain langchain-openai langgraph langchain-core \
            langgraph-checkpoint-sqlite aiosqlite langchain-postgres \
            langchain-community langchain-experimental "psycopg[binary]" \
            pypdf pydantic pydantic-settings python-dotenv tiktoken openai \
            ddgs sentence-transformers fastapi "uvicorn[standard]" sse-starlette \
            langchain-mcp-adapters
cp .env.example .env   # add OPENAI_API_KEY (or configure Ollama)
```

### Infrastructure
```bash
docker compose -f docker/docker-compose.yml up -d
# Postgres (pgvector) on :5434, Grafana on http://localhost:3001

docker exec claude_desktop_pg psql -U claude -d claude_desktop -c "
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE IF EXISTS langchain_pg_embedding
  ADD COLUMN IF NOT EXISTS fts tsvector
  GENERATED ALWAYS AS (to_tsvector('english', document)) STORED;
CREATE INDEX IF NOT EXISTS idx_fts ON langchain_pg_embedding USING GIN (fts);
"
```

### MCP servers (optional but recommended)
```bash
# fetch + filesystem need no install (npx/uvx fetch on first use)
# MemPalace memory server:
uv tool install mempalace
mkdir -p ~/.mempalace
mempalace init ~/.mempalace --yes --no-llm
# then add the servers to mcp_servers.json (see MCP integration section)
```

### Run
```bash
python -m app.rag.ingest_demo      # ingest a document
python -m app.web.server           # launch chat UI at http://localhost:8000
```

---

## Running fully local (Ollama)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
ollama create qwen3-agent -f Modelfile.qwen   # larger context for agent loops
```
Then in `.env`: `LLM_PROVIDER=ollama`, `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=qwen3-agent`. No code changes.

---

## Roadmap

- [x] MCP integration (fetch, filesystem, MemPalace)
- [ ] Tool curation (manage tool count for reliable selection)
- [ ] Obsidian vault as an MCP knowledge base
- [ ] Eval harness (benchmark memory approaches against LongMemEval)
- [ ] Conversation summarization for long threads
- [ ] Containerize the app itself (one-command full stack)
- [ ] UI polish (model picker, markdown rendering, citations as links)

---

## License

MIT