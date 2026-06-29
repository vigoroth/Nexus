"""Phase 2: FastAPI backend with conversation persistence + streaming.

Run:  python -m app.web.server
Then open http://localhost:8000
"""
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.agent.graph import build_graph
from app.web.conversations import (
    init_tables, create_conversation, list_conversations,
    add_message, get_messages,
)

app = FastAPI(title="Nexus")

# open a persistent checkpointer for per-conversation memory
_checkpointer_cm = AsyncSqliteSaver.from_conn_string("data/web_memory.sqlite")
CHECKPOINTER = _checkpointer_cm.__aenter__()

GRAPH = None

async def get_graph():
    global GRAPH, CHECKPOINTER
    if GRAPH is None:
        CHECKPOINTER = await _checkpointer_cm.__aenter__()
        GRAPH = await build_graph(checkpointer=CHECKPOINTER)
    return GRAPH

init_tables()


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


@app.get("/")
def index() -> HTMLResponse:
    html = Path(__file__).parent / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/conversations")
def conversations():
    """List all conversations for the sidebar."""
    return list_conversations()


@app.get("/conversations/{conv_id}")
def conversation_messages(conv_id: str):
    """Return all messages in one conversation, to reload it."""
    return get_messages(conv_id)


@app.post("/chat")
async def chat(req: ChatRequest):
    """Stream the agent's reply; persist both user and assistant messages."""

    # new conversation if none given — title from the first message
    conv_id = req.conversation_id or create_conversation(req.message)

    # save the user's message immediately
    add_message(conv_id, "user", req.message)

    async def event_generator():
        # send the conversation id first so the frontend can track it
        yield {"event": "conversation", "data": conv_id}

        full_reply = ""
        graph = await get_graph()
        config = {"configurable": {"thread_id": conv_id}}
        async for chunk, meta in graph.astream(
            {"messages": [HumanMessage(content=req.message)]},
            stream_mode="messages",
            config=config,
        ):
            content = getattr(chunk, "content", None)
            if content and isinstance(content, str):
                full_reply += content
                yield {"data": json.dumps(content)}

        # save the complete assistant reply once streaming finishes
        add_message(conv_id, "assistant", full_reply)
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()