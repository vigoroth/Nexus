"""Write Nexus conversations to an Obsidian vault as markdown notes.
Each conversation becomes one .md file; Obsidian's graph view visualizes them.
"""
import os
import re
from pathlib import Path
from datetime import datetime

VAULT_PATH = Path(os.environ.get("NEXUS_VAULT_PATH", str(Path.home() / "vault")))
CONV_DIR = VAULT_PATH / "Conversations"


def _safe_filename(title: str) -> str:
    """Turn a conversation title into a safe .md filename."""
    name = re.sub(r'[^\w\s-]', '', title).strip()[:60] or "untitled"
    return name


def write_conversation(conv_id: str, title: str, messages: list[dict]) -> None:
    """Write/overwrite a conversation as a markdown note in the vault.
    messages: list of {role, content}.
    """
    try:
        CONV_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{_safe_filename(title)}.md"
        path = CONV_DIR / fname

        lines = [f"# {title}", f"*{datetime.now():%Y-%m-%d %H:%M}*", ""]
        for m in messages:
            who = "You" if m["role"] == "user" else "Nexus"
            lines.append(f"**{who}:** {m['content']}")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        print(f"vault write failed: {e}")  # don't break chat if vault write fails