"""Phase 2: FastAPI backend with conversation persistence + streaming + auth.

Run:  python -m app.web.server
Then open http://localhost:8000
"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import json
from fastapi import FastAPI, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.agent.graph import build_graph
from app.web.conversations import (
    init_tables, create_conversation, list_conversations,
    add_message, get_messages,
)
from app.web.auth import (
    check_login, make_session_token, valid_session,
    require_auth, COOKIE_NAME, MAX_AGE,
)
from app.web.models_list import list_all_models
app = FastAPI(title="Nexus")

LOGIN_HTML = """
<!doctype html><html><head><title>Nexus — Login</title>
<style>
  body{background:#121212;color:#e8e8e8;font-family:system-ui;display:flex;
       align-items:center;justify-content:center;height:100vh;margin:0}
  form{background:#1e1e1e;padding:32px;border-radius:12px;border:1px solid #333}
  h2{margin:0 0 16px;color:#d97757}
  input{display:block;width:240px;padding:10px;margin:8px 0;background:#252525;
        border:1px solid #444;border-radius:6px;color:#fff}
  button{padding:10px 20px;background:#d97757;border:0;border-radius:6px;
         color:#fff;cursor:pointer;font-weight:600}
  .err{color:#e06c6c;font-size:13px;height:16px}
</style></head><body>
<form method="post" action="/login">
  <h2>Nexus</h2>
  <input type="text" name="username" placeholder="Username" autofocus>
  <input type="password" name="password" placeholder="Password">
  <div class="err">{error}</div>
  <button type="submit">Enter</button>
</form></body></html>
"""






@app.get("/login")
def login_page():
    return HTMLResponse(LOGIN_HTML.replace("{error}", ""))


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if check_login(username, password):
        resp = RedirectResponse(url="/", status_code=303)
        resp.set_cookie(
            COOKIE_NAME, make_session_token(),
            max_age=MAX_AGE, httponly=True, samesite="lax",
        )
        return resp
    return HTMLResponse(LOGIN_HTML.replace("{error}", "Wrong username or password"), status_code=401)

@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    return resp



@app.get("/models", dependencies=[Depends(require_auth)])
async def models():
    """Return available models grouped by provider, for the selector."""
    return await list_all_models()


# persistent async checkpointer for per-conversation memory (built lazily)
_checkpointer_cm = AsyncSqliteSaver.from_conn_string("data/web_memory.sqlite")
CHECKPOINTER = None
GRAPHS = {}  # model_name -> compiled graph


async def get_graph(model: str | None = None, provider: str | None = None):
    global CHECKPOINTER
    if CHECKPOINTER is None:
        CHECKPOINTER = await _checkpointer_cm.__aenter__()
    key = f"{provider or 'default'}:{model or 'default'}"
    if key not in GRAPHS:
        GRAPHS[key] = await build_graph(checkpointer=CHECKPOINTER, model=model, provider=provider)
    return GRAPHS[key]


init_tables()


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    model: str | None = None
    provider: str | None = None


@app.get("/")
def index(request: Request):
    if not valid_session(request):
        return RedirectResponse(url="/login", status_code=303)
    html = Path(__file__).parent / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/conversations", dependencies=[Depends(require_auth)])
def conversations():
    """List all conversations for the sidebar."""
    return list_conversations()


@app.get("/conversations/{conv_id}", dependencies=[Depends(require_auth)])
def conversation_messages(conv_id: str):
    """Return all messages in one conversation, to reload it."""
    return get_messages(conv_id)


@app.post("/chat", dependencies=[Depends(require_auth)])
async def chat(req: ChatRequest):
    """Stream the agent's reply; persist both user and assistant messages."""
    conv_id = req.conversation_id or create_conversation(req.message)
    add_message(conv_id, "user", req.message)

    async def event_generator():
        yield {"event": "conversation", "data": conv_id}
        full_reply = ""
        graph = await get_graph(req.model, req.provider)
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
        add_message(conv_id, "assistant", full_reply)
        # also write the whole conversation to the Obsidian vault
        try:
            from app.web.vault_writer import write_conversation
            from app.web.conversations import list_conversations
            msgs = get_messages(conv_id)
            title = next((c["title"] for c in list_conversations()
                          if c["id"] == conv_id), conv_id)
            write_conversation(conv_id, title, msgs)
        except Exception as e:
            print(f"vault write skipped: {e}")
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()