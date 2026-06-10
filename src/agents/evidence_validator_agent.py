from __future__ import annotations

import json
import re

from src.agents.base import BaseAgent
from src.agents.issue_analyzer_agent import IssueAnalyzerAgent


def evidence_excerpt(item, max_chars=1000):
    content = item.get("expanded_content") or item.get("content") or ""
    content = re.sub(r"\s+", " ", str(content)).strip()

    if len(content) > max_chars:
        return content[:max_chars] + "..."

    return content


def compact_evidence_for_validation(evidence, max_items=10):
    output = []

    for i, item in enumerate(evidence[:max_items], start=1):
        metadata = item.get("metadata") or {}

        output.append(
            {
                "index": i,
                "source_type": item.get("source_type"),
                "title": item.get("title"),
                "context_mode": item.get("context_mode"),
                "source_url": item.get("source_url"),
                "metadata": {
                    "topic_title": metadata.get("topic_title"),
                    "subject_title": metadata.get("subject_title"),
                    "article_title": metadata.get("article_title"),
                    "original_article_number": metadata.get("original_article_number"),
                    "doc_code": metadata.get("doc_code"),
                    "subject": metadata.get("subject"),
                },
                "excerpt": evidence_excerpt(item),
            }
        )

    return output


def normalize_validator_output(data: dict, evidence_count: int) -> dict:
    accepted = []
    rejected = []
    missing = []

    for item in data.get("accepted") or []:
        try:
            index = int(item.get("index"))
        except Exception:
            continue

        if not 1 <= index <= evidence_count:
            continue

        covered_issues = item.get("covered_issues") or []
        covered_issues = [
            str(issue).strip()
            for issue in covered_issues
            if str(issue).strip()
        ]

        accepted.append(
            {
                "index": index,
                "covered_issues": covered_issues,
                "reason": str(item.get("reason") or ""),
            }
        )

    accepted_indexes = {item["index"] for item in accepted}

    for item in data.get("rejected") or []:
        try:
            index = int(item.get("index"))
        except Exception:
            continue

        if not 1 <= index <= evidence_count:
            continue

        if index in accepted_indexes:
            continue

        rejected.append(
            {
                "index": index,
                "reason": str(item.get("reason") or ""),
            }
        )

    for item in data.get("missing") or []:
        hint = str(item.get("hint") or "").strip()
        if not hint:
            continue

        source_type = str(item.get("source_type") or "phapdien").strip().lower()
        if source_type not in {"phapdien", "anle"}:
            source_type = "phapdien"

        missing.append(
            {
                "source_type": source_type,
                "hint": hint,
            }
        )

    return {
        "accepted": accepted,
        "rejected": rejected,
        "missing": missing,
    }


def evidence_identity(item: dict):
    metadata = item.get("metadata") or {}

    return (
        item.get("source_type"),
        metadata.get("parent_uid")
        or metadata.get("article_anchor")
        or metadata.get("doc_name")
        or item.get("source_url")
        or item.get("title"),
    )


def dedupe_evidence(evidence: list) -> list:
    output = []
    seen = set()

    for item in evidence:
        key = evidence_identity(item)
        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


def apply_validation(search_result: dict, validation: dict) -> dict:
    updated = dict(search_result)
    locked_evidence = search_result.get("locked_evidence") or []
    curated_evidence = validation.get("curated_evidence") or []

    updated["original_evidence"] = search_result.get("evidence") or []
    updated["evidence"] = dedupe_evidence(locked_evidence + curated_evidence)
    updated["validation"] = validation
    return updated


class EvidenceValidatorAgent(BaseAgent):
    prompt_id = "evidence_validator"

    def __init__(self, issue_analyzer: IssueAnalyzerAgent | None = None):
        super().__init__()
        self.issue_analyzer = issue_analyzer or IssueAnalyzerAgent()

    def build_prompt(
        self,
        query: str,
        search_result: dict,
        compact_evidence: list,
        issue_analysis: dict,
    ) -> str:
        return self.render_prompt(
            query=query,
            intent=search_result.get("intent"),
            quotas=json.dumps(search_result.get("quotas"), ensure_ascii=False),
            source_route=json.dumps(search_result.get("source_route") or {}, ensure_ascii=False),
            legal_issues=json.dumps(issue_analysis, ensure_ascii=False, indent=2),
            evidence=json.dumps(compact_evidence, ensure_ascii=False, indent=2),
        )

    def run(
        self,
        query: str,
        search_result: dict,
        max_items: int = 10,
        issue_analysis: dict | None = None,
    ) -> dict:
        evidence = search_result.get("evidence") or []
        compact = compact_evidence_for_validation(evidence, max_items=max_items)

        if not compact:
            return {
                "issue_analysis": {
                    "legal_issues": [query],
                    "must_have_source_types": [],
                    "reason": "No evidence to validate.",
                },
                "accepted": [],
                "rejected": [],
                "missing": [],
                "curated_evidence": [],
                "raw": None,
            }

        issue_analysis = issue_analysis or self.issue_analyzer.run(query, search_result)

        try:
            raw_data = self.invoke_json(
                retries=3,
                query=query,
                intent=search_result.get("intent"),
                quotas=json.dumps(search_result.get("quotas"), ensure_ascii=False),
                source_route=json.dumps(search_result.get("source_route") or {}, ensure_ascii=False),
                legal_issues=json.dumps(issue_analysis, ensure_ascii=False, indent=2),
                evidence=json.dumps(compact, ensure_ascii=False, indent=2),
            )
            normalized = normalize_validator_output(raw_data, len(compact))
        except Exception as exc:
            fallback_evidence = evidence[:max_items]
            return {
                "issue_analysis": issue_analysis,
                "accepted": [
                    {
                        "index": i,
                        "covered_issues": [],
                        "reason": "Validator failed; fallback accepted original evidence order.",
                    }
                    for i in range(1, len(fallback_evidence) + 1)
                ],
                "rejected": [],
                "missing": [],
                "curated_evidence": fallback_evidence,
                "raw": {"error": str(exc)},
            }

        accepted_indexes = [item["index"] for item in normalized["accepted"]]
        curated_evidence = [
            evidence[index - 1]
            for index in accepted_indexes
            if 1 <= index <= len(evidence)
        ]

        if not curated_evidence:
            curated_evidence = evidence[:min(4, len(evidence))]

        return {
            "issue_analysis": issue_analysis,
            **normalized,
            "curated_evidence": curated_evidence,
            "raw": raw_data,
        }
