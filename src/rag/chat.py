from __future__ import annotations

from src.agents.answer_generator_agent import AnswerGeneratorAgent
from src.agents.context import evidence_for_chat_prompt, format_memory
from src.agents.memory_rewrite_agent import MemoryRewriteAgent
from src.workflows.legal_rag_workflow import (
    chat,
    extract_article_numbers,
    lookup_articles_from_memory,
    prepair,
    prepare_chat_search,
    prepend_memory_reference_evidence,
    run_legal_rag_chat,
    run_validation_pipeline,
)


def rewrite_query_with_memory(message: str, memory: dict) -> dict:
    return MemoryRewriteAgent().run(message, memory)


def build_chat_prompt(
    original_query: str,
    rewrite_result: dict,
    memory: dict,
    search_result: dict,
) -> str:
    return AnswerGeneratorAgent().build_prompt(
        original_query=original_query,
        rewrite_result=rewrite_result,
        memory=memory,
        search_result=search_result,
    )


__all__ = [
    "build_chat_prompt",
    "chat",
    "evidence_for_chat_prompt",
    "extract_article_numbers",
    "format_memory",
    "lookup_articles_from_memory",
    "prepair",
    "prepare_chat_search",
    "prepend_memory_reference_evidence",
    "rewrite_query_with_memory",
    "run_legal_rag_chat",
    "run_validation_pipeline",
]
