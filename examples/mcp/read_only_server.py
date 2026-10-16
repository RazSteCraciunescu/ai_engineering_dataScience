"""Optional read-only MCP server exposing health, search and RAG tools."""
from __future__ import annotations

from feedback_intelligence.embeddings import search
from feedback_intelligence.rag import answer


def main() -> None:
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("feedback-intelligence")

    @server.tool()
    def semantic_search(query: str, limit: int = 3) -> list[dict[str, object]]:
        """Search local documentation without modifying repository data."""
        return search(query, limit=limit)

    @server.tool()
    def grounded_answer(question: str, limit: int = 3) -> dict[str, object]:
        """Return a deterministic answer and its local evidence."""
        return answer(question, limit=limit)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
