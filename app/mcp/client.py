"""Load tools from configured MCP servers (config-driven, resilient)."""
import json
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

CONFIG_PATH = Path(__file__).parent.parent.parent / "mcp_servers.json"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"WARNING: {CONFIG_PATH} not found")
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception as e:
        print(f"WARNING: could not read {CONFIG_PATH}: {e}")
        return {}


async def load_mcp_tools() -> list:
    servers = _load_config()
    if not servers:
        return []
    all_tools = []
    for name, cfg in servers.items():
        try:
            client = MultiServerMCPClient({name: cfg})
            tools = await client.get_tools()
            print(f"MCP '{name}': loaded {len(tools)} tools")
            all_tools.extend(tools)
        except Exception as e:
            print(f"MCP '{name}' FAILED: {type(e).__name__}: {str(e)[:200]}")
    return all_tools
