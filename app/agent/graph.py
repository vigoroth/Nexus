from app.mcp.client import load_mcp_tools
from app.tools.web_search import web_search
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from app.tools.os_tools import run_shell
from app.agent.state import AgentState
from app.core.llm import get_llm
from app.tools.os_tools import OS_TOOLS
from app.tools.rag_tool import search_documents
from langgraph.checkpoint.sqlite import SqliteSaver
from app.tools.memory_tools import save_memory, load_memory
from dotenv import load_dotenv
from app.mcp.client import load_mcp_tools




load_dotenv()


SYSTEM_PROMPT = """You are a helpful personal AI assistant named Friday with access to tools.

Tool selection rules:
- For anything about the user's OWN notes, documents, saved advice, or "my
  documents/files", use search_documents. This is a vector search over their
  ingested knowledge base, not the filesystem.
- Use read_file / list_dir / run_shell only for actual filesystem paths the
  user explicitly names.

- To read the full contents of a specific web page or URL, use the fetch tool (it retrieves and extracts page content as markdown).
 Use web_search to find pages by topic; use fetch to read a URL you already have.

Think step by step. Be concise. When you answer from search_documents results,
cite the source numbers like [1], [2].

-To retrieve the full contents of a specific URL or web page, use the fetch tool.
    Use web_search to find pages; use fetch to read a known URL.

-For current events, recent news, prices, or facts that may have changed,
 use web_search. For the user's own documents use search_documents.

-MEMORY — THIS IS A STANDING INSTRUCTION, NOT OPTIONAL:
The MOMENT the user states any personal fact about themselves — name, location,
job, a pet, a preference, a project, a relationship, a goal, a date, anything
they'd expect you to recall later — you MUST call save_memory immediately,
BEFORE or ALONGSIDE your conversational reply. This applies even when the fact
is mentioned casually, in passing, or as an aside ("by the way...", "I just...").
Casual phrasing does NOT mean it's unimportant. Saving is part of every response,
not a separate task you choose to do.

Use a short snake_case key and the exact value, e.g.
save_memory(key="pet_cat", value="Mochi") for "I adopted a cat named Mochi".

Do NOT save one-off task details, trivia, or anything not about the user.
The facts you already know are listed below — don't re-save those."""


async def build_graph(checkpointer=None, model: str | None = None,
                      provider: str | None = None,
                      memory_backend: str = "both"):
        """memory_backend: 'postgres' | 'mempalace' | 'both'"""
        mcp_tools = await load_mcp_tools()

        # split MCP tools: separate MemPalace memory tools from the rest
        mempalace_mem = [t for t in mcp_tools if t.name in
                        ("mempalace_add_drawer", "mempalace_search")]
        other_mcp = [t for t in mcp_tools if not t.name.startswith("mempalace_")]
        mempalace_other = [t for t in mcp_tools
                        if t.name.startswith("mempalace_") and t not in mempalace_mem]

        # base tools (always present)
        base = [search_documents, web_search, run_shell] + other_mcp + mempalace_other

        # memory tools depend on the backend under test
        if memory_backend == "postgres":
            mem_tools = [save_memory, load_memory]
        elif memory_backend == "mempalace":
            mem_tools = mempalace_mem
        else:  # both (normal operation)
            mem_tools = [save_memory, load_memory] + mempalace_mem

        all_tools = base + mem_tools
        llm = get_llm(streaming=True, model=model, provider=provider).bind_tools(all_tools)
        
        def llm_node(state: AgentState) -> dict:
                        from app.memory.long_term import recall_all
                        facts = recall_all()
                        prompt = SYSTEM_PROMPT
                        if facts:
                            known = "\n".join(f"- {k}: {v}" for k, v in facts.items())
                            prompt += (
                                "\n\n[INTERNAL CONTEXT — do NOT list, repeat, echo, or "
                                "recite this block to the user. Use these facts only to "
                                "answer naturally when directly relevant.]\n"
                                "Known facts about the user:\n" + known
                            )
                        messages = [SystemMessage(content=prompt)] + state["messages"]
                        response = llm.invoke(messages)
                        return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            """Decide: loop back to tools, or stop and return the answer."""
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "end"

        # ToolNode handles running the actual tool functions and
        # returning ToolMessage results back into state
        tool_node = ToolNode(all_tools)

        graph = StateGraph(AgentState)

        graph.add_node("llm", llm_node)
        graph.add_node("tools", tool_node)

        graph.add_edge(START, "llm")

        graph.add_conditional_edges(
            "llm",
            should_continue,
            {"tools": "tools", "end": END},
        )

        graph.add_edge("tools", "llm")

        return graph.compile(checkpointer=checkpointer)