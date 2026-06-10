from __future__ import annotations

import re
import sqlite3

from src.agents.answer_generator_agent import AnswerGeneratorAgent
from src.agents.memory_rewrite_agent import MemoryRewriteAgent
from src.agents.phrase_extractor_agent import PhraseExtractorAgent
from src.agents.source_router_agent import SourceRouterAgent
from src.memory.conversation import load_memory, save_turn
from src.retrieval.repair import repair_missing_evidence
from src.retrieval.ontology_router import route_query
from src.retrieval.retriever import search_all
from src.retrieval.validator import apply_validation, validate_evidence

REFERENCE_DB = "data/index/phapdien_reference_index.sqlite"


def extract_article_numbers(text: str) -> list:
    if not text:
        return []

    numbers = re.findall(r"(?:Điều|Dieu)\s+(\d+)", text, flags=re.IGNORECASE)
    output = []
    seen = set()

    for number in numbers:
        if number in seen:
            continue

        seen.add(number)
        output.append(number)

    return output


def lookup_articles_from_memory(original_query: str, rewrite_result: dict, memory: dict) -> list:
    active_context = memory.get("active_context") or {}
    subject_title = active_context.get("subject_title")

    if not subject_title:
        return []

    search_text = " ".join(
        [
            original_query or "",
            rewrite_result.get("rewritten_query") or "",
        ]
    )
    article_numbers = extract_article_numbers(search_text)

    if not article_numbers:
        return []

    conn = sqlite3.connect(REFERENCE_DB)
    conn.row_factory = sqlite3.Row

    output = []

    try:
        for article_number in article_numbers:
            row = conn.execute(
                """
                SELECT *
                FROM articles
                WHERE subject_title = ?
                  AND original_article_number = ?
                ORDER BY
                  CASE WHEN article_title LIKE '%.LQ.%' THEN 0 ELSE 1 END,
                  LENGTH(COALESCE(content_text, '')) DESC
                LIMIT 1
                """,
                (subject_title, article_number),
            ).fetchone()

            if row is None:
                continue

            article = dict(row)
            content = article.get("content_text") or ""

            output.append(
                {
                    "source_type": "phapdien",
                    "title": article.get("article_title"),
                    "content": content,
                    "expanded_content": content,
                    "source_url": article.get("source_url"),
                    "score": 0,
                    "retrieval_mode": "memory_reference",
                    "context_mode": "full_article",
                    "metadata": article,
                }
            )
    finally:
        conn.close()

    return output


def prepend_memory_reference_evidence(search_result: dict, references: list) -> dict:
    if not references:
        return search_result

    existing = search_result.get("evidence") or []
    seen = {
        (
            item.get("source_type"),
            item.get("source_url") or item.get("title"),
        )
        for item in references
    }

    evidence = list(references)

    for item in existing:
        key = (
            item.get("source_type"),
            item.get("source_url") or item.get("title"),
        )
        if key in seen:
            continue

        seen.add(key)
        evidence.append(item)

    updated = dict(search_result)
    updated["evidence"] = evidence
    return updated


def run_validation_pipeline(
    query: str,
    search_result: dict,
    validation_mode: str = "none",
) -> tuple[dict, dict]:
    if validation_mode not in {"none", "validate", "repair"}:
        validation_mode = "none"

    debug = {
        "validation_mode": validation_mode,
        "validation_1": None,
        "validation_2": None,
        "repair": None,
    }

    if validation_mode == "none":
        return search_result, debug

    validation_1 = validate_evidence(query, search_result, max_items=8)
    debug["validation_1"] = validation_1

    validated_result = apply_validation(search_result, validation_1)

    if validation_mode == "validate":
        return validated_result, debug

    repair = repair_missing_evidence(
        query=query,
        search_result=validated_result,
        validation=validation_1,
        k_per_hint=5,
        max_hints=5,
    )

    debug["repair"] = repair

    repaired_result = repair["search_result"]

    validation_2 = validate_evidence(query, repaired_result, max_items=12)
    debug["validation_2"] = validation_2

    final_result = apply_validation(repaired_result, validation_2)

    return final_result, debug


def prepare_chat_search(session_id: str, message: str, k: int = 8, validation_mode: str = "none") -> dict:
    memory = load_memory(session_id)
    rewrite_result = MemoryRewriteAgent().run(message, memory)
    rewritten_query = rewrite_result["rewritten_query"]

    route = route_query(rewritten_query)
    source_route = SourceRouterAgent().run(rewritten_query, total_k=k)
    phrases = PhraseExtractorAgent().run(rewritten_query, route=route)

    search_result = search_all(
        rewritten_query,
        k=k,
        route=route,
        source_route=source_route,
        phrases=phrases,
    )
    memory_references = lookup_articles_from_memory(message, rewrite_result, memory)
    search_result = prepend_memory_reference_evidence(search_result, memory_references)

    search_result, validation_debug = run_validation_pipeline(
        query=rewritten_query,
        search_result=search_result,
        validation_mode=validation_mode,
    )

    return {
        "session_id": session_id,
        "message": message,
        "memory": memory,
        "rewrite_result": rewrite_result,
        "rewritten_query": rewritten_query,
        "search_result": search_result,
        "validation_debug": validation_debug,
    }


def prepair(session_id: str, message: str, k: int = 8, validation_mode: str = "none") -> dict:
    return prepare_chat_search(
        session_id=session_id,
        message=message,
        k=k,
        validation_mode=validation_mode,
    )


def run_legal_rag_chat(session_id: str, message: str, k: int = 8, validation_mode: str = "none") -> dict:
    prepared = prepare_chat_search(
        session_id=session_id,
        message=message,
        k=k,
        validation_mode=validation_mode,
    )

    memory = prepared["memory"]
    rewrite_result = prepared["rewrite_result"]
    rewritten_query = prepared["rewritten_query"]
    search_result = prepared["search_result"]
    validation_debug = prepared["validation_debug"]

    answer_result = AnswerGeneratorAgent().run(
        original_query=message,
        rewrite_result=rewrite_result,
        memory=memory,
        search_result=search_result,
    )

    save_turn(
        session_id=session_id,
        user_message=message,
        assistant_answer=answer_result["answer"],
        search_result=search_result,
    )

    return {
        "answer": answer_result["answer"],
        "session_id": session_id,
        "original_query": message,
        "rewritten_query": rewritten_query,
        "rewrite": rewrite_result,
        "intent": search_result["intent"],
        "quotas": search_result["quotas"],
        "route": search_result.get("route"),
        "source_route": search_result.get("source_route"),
        "query_phrases": search_result.get("query_phrases"),
        "evidence": search_result["evidence"],
        "validation_mode": validation_mode,
        "validation_1": validation_debug.get("validation_1"),
        "validation_2": validation_debug.get("validation_2"),
        "repair": validation_debug.get("repair"),
        "validation": search_result.get("validation"),
        "original_evidence": search_result.get("original_evidence"),
    }


def chat(session_id: str, message: str, k: int = 8, validation_mode: str = "none") -> dict:
    return run_legal_rag_chat(
        session_id=session_id,
        message=message,
        k=k,
        validation_mode=validation_mode,
    )

