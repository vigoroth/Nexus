from langchain_core.tools import tool

from app.rag.retriever import search


@tool
def search_documents(query: str) -> str:
    """Search the user's personal knowledge base of ingested documents.
    ALWAYS use this tool first when the user asks about their notes, their
    documents, advice they've saved, or anything in "my documents" / "my files".
    This searches an indexed vector database, NOT the filesystem. Do not use
    shell or file-reading tools to look for ingested content; use this instead.
    """
    results = search(query, k=4)
    if not results:
        return "No relevant documents found."

    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"[{i}] (source: {r['source']}, score: {r['score']})\n{r['text']}"
        )
    return "\n\n".join(blocks)