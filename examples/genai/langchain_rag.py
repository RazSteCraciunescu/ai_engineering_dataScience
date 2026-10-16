"""Optional LangChain composition around the repository's local retriever."""
from __future__ import annotations

from feedback_intelligence.embeddings import search


def main() -> None:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    question = "What is data leakage?"
    context = "\n\n".join(item["text"] for item in search(question, limit=3))
    prompt = ChatPromptTemplate.from_template(
        "Use only the context. If it is insufficient, say so.\nContext:\n{context}\nQuestion: {question}"
    )
    chain = prompt | ChatOllama(model="llama3.2:3b", temperature=0) | StrOutputParser()
    print(chain.invoke({"context": context, "question": question}))


if __name__ == "__main__":
    main()
