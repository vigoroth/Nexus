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
 use web_search. For the user's own documents use search_documents; 
 for their personal facts use load_memory."""


async def build_graph(checkpointer=None) -> StateGraph:
        """Build and compile the agent graph."""
        mcp_tools = await load_mcp_tools()
        # bind tools to the LLM so it knows what it can call
        all_tools =[search_documents, save_memory, load_memory, web_search, run_shell] + mcp_tools
        llm = get_llm(streaming=True).bind_tools(all_tools)

        def llm_node(state: AgentState) -> dict:
            """Call the LLM with the current message history."""
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
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