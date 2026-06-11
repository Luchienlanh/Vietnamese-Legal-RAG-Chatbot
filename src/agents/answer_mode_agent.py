from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.agents.context import format_memory


ALLOWED_MODES = {
    "normal",
    "summary",
    "full_provision",
    "full_case",
    "detail_case",
    "compare",
    "procedure",
}
ALLOWED_TARGET_SOURCES = {"any", "phapdien", "anle", "vanban"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}


def as_clean_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]

    output = []
    seen = set()

    for item in values:
        text = str(item).strip()

        if not text or text in seen:
            continue

        seen.add(text)
        output.append(text)

    return output


def default_answer_mode(reason: str) -> dict[str, Any]:
    return {
        "mode": "normal",
        "target_source": "any",
        "article_numbers": [],
        "clause_numbers": [],
        "case_numbers": [],
        "requires_exact_text": False,
        "requires_full_document": False,
        "uses_reference": False,
        "confidence": "low",
        "reason": reason,
        "wants_summary": False,
        "wants_full_text": False,
        "wants_detail": False,
    }


def normalize_answer_mode(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return default_answer_mode("Answer mode classifier returned a non-object value.")

    mode = str(raw.get("mode") or "normal").strip().lower()
    if mode not in ALLOWED_MODES:
        mode = "normal"

    target_source = str(raw.get("target_source") or "any").strip().lower()
    if target_source not in ALLOWED_TARGET_SOURCES:
        target_source = "any"

    confidence = str(raw.get("confidence") or "low").strip().lower()
    if confidence not in ALLOWED_CONFIDENCE:
        confidence = "low"

    requires_exact_text = bool(raw.get("requires_exact_text"))
    requires_full_document = bool(raw.get("requires_full_document"))

    if mode == "full_provision":
        requires_exact_text = True
        target_source = "phapdien" if target_source == "any" else target_source
    elif mode == "full_case":
        requires_exact_text = True
        requires_full_document = True
        target_source = "anle" if target_source == "any" else target_source
    elif mode == "detail_case":
        target_source = "anle" if target_source == "any" else target_source

    return {
        "mode": mode,
        "target_source": target_source,
        "article_numbers": as_clean_list(raw.get("article_numbers")),
        "clause_numbers": as_clean_list(raw.get("clause_numbers")),
        "case_numbers": as_clean_list(raw.get("case_numbers")),
        "requires_exact_text": requires_exact_text,
        "requires_full_document": requires_full_document,
        "uses_reference": bool(raw.get("uses_reference")),
        "confidence": confidence,
        "reason": str(raw.get("reason") or "").strip(),
        "wants_summary": mode == "summary",
        "wants_full_text": mode in {"full_provision", "full_case"},
        "wants_detail": mode == "detail_case",
    }


class AnswerModeAgent(BaseAgent):
    prompt_id = "answer_mode"

    def run(
        self,
        original_query: str,
        rewritten_query: str | None = None,
        memory: dict | None = None,
    ) -> dict[str, Any]:
        try:
            result = self.invoke_json(
                retries=3,
                original_query=original_query,
                rewritten_query=rewritten_query or original_query,
                memory=format_memory(memory or {}),
            )
        except Exception as exc:
            return default_answer_mode(f"Answer mode classifier failed: {exc}")

        return normalize_answer_mode(result)
