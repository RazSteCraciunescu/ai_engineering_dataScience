"""Optional Langfuse tracing adapter; requires user-provided environment configuration."""
from __future__ import annotations

from feedback_intelligence.rag import answer


def main() -> None:
    from langfuse import Langfuse

    client = Langfuse()
    question = "What is the difference between normalization and standardization?"
    trace = client.trace(name="local-rag", input={"question": question})
    result = answer(question, limit=3)
    trace.update(output=result)
    client.flush()
    print(result["answer"])


if __name__ == "__main__":
    main()
