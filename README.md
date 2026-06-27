# Nexus

A self-hosted, local-first AI agent workspace built from scratch — chat, tool-using agents, retrieval-augmented generation (RAG), persistent memory, and full observability. Inspired by the local-first philosophy of projects like Odysseus, Nexus is designed so you understand every layer, run it on your own hardware, and keep your data yours.

**Version 1.0** — core agent, RAG, memory, and monitoring complete.

---

## What it does

Nexus is an autonomous AI agent that can:

- **Reason and act in a loop** — using the ReAct pattern, it decides when to call tools, reads the results, and continues until it can answer.
- **Operate on your system** — read files, write files, list directories, and run shell commands.
- **Answer from your own documents** — ingest `.txt`, `.md`, and `.pdf` files, chunk and embed them, and retrieve relevant passages with citations (RAG).
- **Remember you** — short-term conversation memory (within a session) and long-term memory (durable facts that persist across every session).
- **Monitor itself** — every run records its latency, token usage, cost, and success/failure to a database, surfaced in a live Grafana dashboard.
- **Swap models freely** — built on an OpenAI-compatible provider layer, so switching from OpenAI to a local model (Ollama, vLLM, LM Studio) is a configuration change, not a code change.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Agent core: planner — tool loop — memory        │
│     ├─ Tools: read/write file, list, shell, RAG  │
│     ├─ LLM provider layer (OpenAI ⇄ local)       │
│     └─ Prompt + context management                │
├─────────────────────────────────────────────────┤
│  RAG: ingest → chunk → embed → pgvector → search │
├─────────────────────────────────────────────────┤
│  Memory: short-term (SQLite) + long-term (PG)    │
├─────────────────────────────────────────────────┤
│  Observability: per-run metrics → Postgres →     │
│                 Grafana dashboards                │
├─────────────────────────────────────────────────┤
│  Infra: Docker Compose (Postgres/pgvector +      │
│         Grafana)                                  │
└─────────────────────────────────────────────────┘
```

---

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.11 | Standard for the AI/LLM ecosystem |
| Agent framework | LangChain + LangGraph | Industry-standard agent orchestration; LangGraph for the stateful tool loop |
| LLM provider | OpenAI (swappable) | OpenAI-compatible layer enables local models later |
| Vector store | Postgres + pgvector | One database for vectors, metadata, and app state; production-common pattern |
| Embeddings | OpenAI `text-embedding-3-small` | Cost-effective, strong retrieval quality |
| Short-term memory | LangGraph SQLite checkpointer | Conversation state persisted per thread |
| Long-term memory | Postgres key-value table | Durable cross-session facts |
| Tracing (dev) | LangSmith | Full execution traces during development |
| Metrics (prod) | Custom recorder → Postgres → Grafana | Local-first observability, no external telemetry |
| Infrastructure | Docker Compose | Postgres + Grafana with one command |

---

## Project structure

```
nexus/
├── app/
│   ├── core/
│   │   ├── config.py            # typed settings, loaded once from .env
│   │   ├── llm.py               # swappable LLM provider + tracked calls
│   │   ├── pricing.py           # token → cost conversion
│   │   └── metrics.py           # per-run metrics recorder
│   ├── tools/
│   │   ├── os_tools.py          # read/write file, list dir, run shell
│   │   ├── rag_tool.py          # document search as an agent tool
│   │   └── memory_tools.py      # save/load long-term memory
│   ├── agent/
│   │   ├── state.py             # the graph state (message history)
│   │   ├── graph.py             # the LangGraph agent loop
│   │   └── agent_demo.py        # run the agent
│   ├── rag/
│   │   ├── loader.py            # load documents into text
│   │   ├── chunker.py           # recursive chunking with overlap
│   │   ├── store.py             # embed + store in pgvector
│   │   ├── retriever.py         # similarity search
│   │   └── ingest_demo.py       # ingestion pipeline
│   └── memory/
│       ├── long_term.py         # Postgres-backed durable memory
│       └── memory_demo.py       # short + long-term memory demos
├── docker/
│   ├── docker-compose.yml       # Postgres/pgvector + Grafana
│   └── grafana/provisioning/    # auto-provisioned data source
├── .env.example                 # configuration template
├── pyproject.toml               # dependencies
└── README.md
```

---

## Quick start

### Prerequisites

- Python 3.11+
- Docker (with Docker Desktop + WSL integration on Windows)
- An OpenAI API key

### 1. Clone and set up the environment

```bash
git clone https://github.com/<your-username>/nexus.git
cd nexus

# create the environment (conda example)
conda create -n nexus python=3.11 -y
conda activate nexus

# install dependencies
pip install langchain langchain-openai langgraph langchain-core \
            langgraph-checkpoint-sqlite langchain-postgres langchain-community \
            "psycopg[binary]" pypdf pydantic pydantic-settings \
            python-dotenv tiktoken openai
```

### 2. Configure secrets

```bash
cp .env.example .env
# edit .env and add your OPENAI_API_KEY
```

### 3. Start the infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d
# Postgres (pgvector) on localhost:5434
# Grafana on http://localhost:3001  (admin / admin)
```

### 4. Run it

```bash
# ingest a document into the vector store
python -m app.rag.ingest_demo

# run the agent (uses tools, RAG, and memory)
python -m app.agent.agent_demo

# RAG demo — the agent answers from your documents
python -m app.rag.rag_demo

# memory demos — short-term and long-term recall
python -m app.memory.memory_demo
python -m app.memory.long_term_demo
```

---

## Configuration

All configuration lives in `.env` (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | _(required)_ |
| `LLM_MODEL` | Chat model | `gpt-4o-mini` |
| `EMBED_MODEL` | Embedding model | `text-embedding-3-small` |
| `LLM_PROVIDER` | Provider name | `openai` |
| `LLM_BASE_URL` | Override for local/compatible servers | _(blank = OpenAI)_ |
| `DATABASE_URL` | Postgres connection string | `postgresql+psycopg://...localhost:5434...` |

### Switching to a local model

Nexus uses an OpenAI-compatible provider layer. To run against a local model (e.g. Ollama):

```bash
# in .env
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1
```

No code changes required.

---

## Monitoring

Every agent run records its metrics (duration, tokens, cost, success) to the `run_metrics` table in Postgres. A Grafana dashboard (at `http://localhost:3001`) visualizes:

- Cost per run over time
- Total spend
- Run count
- p95 latency
- Error rate
- Token usage

The Postgres data source is auto-provisioned via `docker/grafana/provisioning/`.

---

## Roadmap

- [ ] Containerize the application itself (full `docker compose up` stack)
- [ ] Web chat UI
- [ ] Local model support via Ollama (provider layer is ready)
- [ ] Semantic and late chunking strategies for RAG
- [ ] Hybrid search (dense + keyword) with reranking
- [ ] Multi-agent workflows

---

## Acknowledgements

Built as a from-scratch learning project to deeply understand AI agents, RAG, and observability. Inspired by the local-first, privacy-first philosophy of self-hosted AI workspaces.

## License

MIT