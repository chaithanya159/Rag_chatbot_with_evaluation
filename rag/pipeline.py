"""
rag/pipeline.py
----------------
LLM setup (Groq, free tier — swapped in from Claude) and the core
rag_advanced() function: retrieve context, build a grounded prompt, call
the LLM, return answer + sources + a confidence score.
"""

import os
import re
from typing import Optional

from groq import Groq

from rag.retriever import RAGRetriever

# Answer model on Groq. Groq's available models change: list the ones your key can use
# (see README) and edit this line if this one is not available to you.
GROQ_MODEL = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a specific document.
Answer ONLY using the information in the provided context below.
If the answer cannot be found in the context, say clearly that the document \
does not contain that information — do not make anything up.
Do not use markdown formatting of any kind — no asterisks, no bold, no italics, \
no headers. Respond in plain text only.
When your answer includes multiple distinct items (such as types, steps, or \
categories), put each item on its own line in the format "1. Name: description", \
with a line break between each numbered item, so they can be read as a list.
Keep answers concise, and mention page numbers from the context if available."""


def get_llm(api_key: Optional[str] = None) -> Groq:
    """Creates the Groq client. Reads GROQ_API_KEY from env if not passed."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not found. Set it in .env or the environment.")
    return Groq(api_key=key)


def _strip_markdown(text: str) -> str:
    """
    Removes common markdown formatting from the model's response, in case it
    doesn't fully follow the "plain text only" system instruction. Guarantees
    clean output regardless of model compliance.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)          # **bold**
    text = re.sub(r"__(.+?)__", r"\1", text)                # __bold__
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)  # *italic*
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)          # # headers
    text = re.sub(r"`([^`]+)`", r"\1", text)                # `code`
    return text


def _build_context(results) -> str:
    blocks = []
    for i, (doc, score) in enumerate(results, start=1):
        page_info = f", page {doc.metadata.get('page')}" if doc.metadata.get("page") else ""
        blocks.append(f"[Source {i} | score={score:.2f}{page_info}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _build_history_messages(chat_history: list) -> list:
    """
    Converts recent stored turns into OpenAI-style message turns, so the
    model can resolve follow-ups like "what about the second point?".
    Keeps only question/answer text — not sources — to keep the prompt lean.
    """
    messages = []
    for turn in chat_history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    return messages


def rag_advanced(
    query: str,
    retriever: RAGRetriever,
    llm: Groq,
    top_k: int = 5,
    min_score: float = 0.2,
    return_context: bool = False,
    chat_history: Optional[list] = None,
) -> dict:
    """
    Full RAG query: retrieve -> augment -> generate.
    chat_history: optional list of recent {"question", "answer"} dicts (oldest
    first) used as conversational context for follow-up questions.
    Returns: {"answer": str, "sources": [...], "confidence": float, "context": str (optional)}
    """
    results = retriever.retrieve(query, top_k=top_k, min_score=min_score)

    if not results:
        return {
            "answer": "I couldn't find anything relevant to that question in the document.",
            "sources": [],
            "confidence": 0.0,
        }

    context = _build_context(results)
    user_message = f"""Context from the document:

{context}

Question: {query}"""

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    conversation.extend(_build_history_messages(chat_history or []))
    conversation.append({"role": "user", "content": user_message})

    response = llm.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1024,
        messages=conversation,
        # Qwen3.8 thinks before answering by default; turn that off for document Q&A.
        # If you switch to a model that does not support this option, delete this line.
        extra_body={"reasoning_effort": "none"},
    )
    answer_text = response.choices[0].message.content or ""
    answer_text = _strip_markdown(answer_text)

    sources = [
        {
            "text": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
            "score": round(score, 3),
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
        }
        for doc, score in results
    ]
    confidence = round(max(score for _, score in results), 3)

    output = {"answer": answer_text, "sources": sources, "confidence": confidence}
    if return_context:
        output["context"] = context
    return output
