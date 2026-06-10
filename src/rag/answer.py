from __future__ import annotations

from src.agents.answer_generator_agent import (
    AnswerGeneratorAgent,
    build_answer_policy,
    format_evidence,
)
from src.agents.base import invoke_with_retry
from src.retrieval.retriever import search_all


def build_prompt(query, search_result):
    return AnswerGeneratorAgent().build_prompt(
        original_query=query,
        rewrite_result={"rewritten_query": query},
        memory={},
        search_result=search_result,
    )


def answer_query(query, k=8):
    search_results = search_all(query, k)
    answer_result = AnswerGeneratorAgent().run(
        original_query=query,
        rewrite_result={"rewritten_query": query},
        memory={},
        search_result=search_results,
    )

    return {
        "answer": answer_result["answer"],
        "intent": search_results["intent"],
        "quotas": search_results["quotas"],
        "route": search_results.get("route"),
        "source_route": search_results.get("source_route"),
        "query_phrases": search_results.get("query_phrases"),
        "evidence": search_results["evidence"],
    }
