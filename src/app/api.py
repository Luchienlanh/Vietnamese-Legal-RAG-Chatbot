import math
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.memory.conversation import load_memory, reset_session
from src.rag.chat import chat, prepare_chat_search



app = FastAPI(title="Law RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    k: int = Field(default=8, ge=1, le=20)
    validation_mode: Literal["none", "validate", "repair"] = "none"
    include_debug: bool = False


class ResetResponse(BaseModel):
    session_id: str
    reset: bool


def safe_json(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None

    if isinstance(value, dict):
        return {
            str(key): safe_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            safe_json(item)
            for item in value
        ]

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return safe_json(value.item())
    except Exception:
        pass

    return str(value)


def compact_source(item: dict) -> dict:
    metadata = item.get("metadata") or {}

    return safe_json(
        {
            "source_type": item.get("source_type"),
            "title": item.get("title"),
            "source_url": item.get("source_url"),
            "content": item.get("content") or item.get("expanded_content"),
            "score": item.get("score"),
            "context_mode": item.get("context_mode"),
            "retrieval_mode": item.get("retrieval_mode"),
            "metadata": {
                "topic_title": metadata.get("topic_title"),
                "subject_title": metadata.get("subject_title"),
                "article_title": metadata.get("article_title"),
                "original_article_number": metadata.get("original_article_number"),
                "doc_name": metadata.get("doc_name"),
                "doc_code": metadata.get("doc_code"),
                "subject": metadata.get("subject"),
                "title": metadata.get("title"),
                "detail_url": metadata.get("detail_url"),
                "pdf_url": metadata.get("pdf_url"),
                "source_note_text": metadata.get("source_note_text"),
            },
        }
    )


def compact_sources(evidence: list[dict]) -> list[dict]:
    return [
        compact_source(item)
        for item in evidence
    ]


def compact_validation(validation: dict | None) -> dict | None:
    if not validation:
        return None

    return safe_json(
        {
            "issue_analysis": validation.get("issue_analysis"),
            "accepted": validation.get("accepted"),
            "rejected": validation.get("rejected"),
            "missing": validation.get("missing"),
        }
    )


def compact_repair(repair: dict | None) -> dict | None:
    if not repair:
        return None

    return safe_json(
        {
            "repaired": repair.get("repaired"),
            "missing": repair.get("missing"),
            "added_count": len(repair.get("added_evidence") or []),
            "added_sources": compact_sources(repair.get("added_evidence") or []),
        }
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat_endpoint(request: ChatRequest) -> dict:
    session_id = request.session_id or str(uuid4())

    result = chat(
        session_id=session_id,
        message=request.message,
        k=request.k,
        validation_mode=request.validation_mode,
    )

    response = {
        "session_id": session_id,
        "answer": result["answer"],
        "original_query": result.get("original_query"),
        "rewritten_query": result.get("rewritten_query"),
        "rewrite": result.get("rewrite"),
        "intent": result.get("intent"),
        "answer_mode": result.get("answer_mode"),
        "quotas": result.get("quotas"),
        "validation_mode": result.get("validation_mode"),
        "sources": compact_sources(result.get("evidence") or []),
    }

    if request.include_debug:
        response["debug"] = safe_json(
            {
                "route": result.get("route"),
                "source_route": result.get("source_route"),
                "answer_mode": result.get("answer_mode"),
                "query_phrases": result.get("query_phrases"),
                "validation_1": compact_validation(result.get("validation_1")),
                "validation_2": compact_validation(result.get("validation_2")),
                "validation": compact_validation(result.get("validation")),
                "repair": compact_repair(result.get("repair")),
                "original_sources": compact_sources(result.get("original_evidence") or []),
            }
        )

    return safe_json(response)

@app.post("/search")
def search_endpoint(request: ChatRequest) -> dict:
    session_id = request.session_id or str(uuid4())
    
    prepared = prepare_chat_search(
        session_id=session_id,
        message=request.message,
        k=request.k,
        validation_mode=request.validation_mode,
    )
    
    search_result = prepared["search_result"]
    validation_debug = prepared["validation_debug"]
    
    response = {
        "session_id": session_id,
        "original_query": prepared["message"],
        "rewritten_query": prepared["rewritten_query"],
        "rewrite": prepared["rewrite_result"],
        "intent": search_result.get("intent"),
        "answer_mode": search_result.get("answer_mode"),
        "quotas": search_result.get("quotas"),
        "validation_mode": request.validation_mode,
        "sources": compact_sources(search_result.get("evidence") or []),
    }
    
    if request.include_debug:
        response["debug"] = safe_json({
            "route": search_result.get("route"),
            "source_route": search_result.get("source_route"),
            "answer_mode": search_result.get("answer_mode"),
            "query_phrases": search_result.get("query_phrases"),
            "validation_1": compact_validation(validation_debug.get("validation_1")),
            "validation_2": compact_validation(validation_debug.get("validation_2")),
            "validation": compact_validation(search_result.get("validation")),
            "repair": compact_repair(validation_debug.get("repair")),
            "original_sources": compact_sources(search_result.get("original_evidence") or []),
        })
        
    return safe_json(response)

@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    return safe_json(load_memory(session_id))


@app.post("/sessions/{session_id}/reset", response_model=ResetResponse)
def reset_session_endpoint(session_id: str) -> ResetResponse:
    reset_session(session_id)
    return ResetResponse(session_id=session_id, reset=True)

