from __future__ import annotations

import json

from src.agents.base import BaseAgent


def normalize_issue_analysis(data: dict, query: str) -> dict:
    issues = data.get("legal_issues") or []
    normalized_issues = []
    seen = set()

    for issue in issues:
        issue = str(issue).strip()
        if not issue:
            continue

        key = issue.lower()
        if key in seen:
            continue

        seen.add(key)
        normalized_issues.append(issue)

    if not normalized_issues:
        normalized_issues = [query]

    source_types = data.get("must_have_source_types") or []
    normalized_source_types = []

    for source_type in source_types:
        source_type = str(source_type).strip().lower()
        if source_type in {"phapdien", "anle"}:
            normalized_source_types.append(source_type)

    return {
        "legal_issues": normalized_issues[:8],
        "must_have_source_types": normalized_source_types[:3],
        "reason": str(data.get("reason") or ""),
    }


class IssueAnalyzerAgent(BaseAgent):
    prompt_id = "issue_analyzer"

    def build_prompt(self, query: str, search_result: dict) -> str:
        route = search_result.get("route") or {}
        topic = (route.get("topic") or {}).get("topic_title") or ""
        subject = (route.get("subject") or {}).get("subject_title") or ""
        source_route = search_result.get("source_route") or {}

        return self.render_prompt(
            query=query,
            topic=topic,
            subject=subject,
            source_route=json.dumps(source_route, ensure_ascii=False),
        )

    def run(self, query: str, search_result: dict) -> dict:
        route = search_result.get("route") or {}
        topic = (route.get("topic") or {}).get("topic_title") or ""
        subject = (route.get("subject") or {}).get("subject_title") or ""
        source_route = search_result.get("source_route") or {}

        try:
            data = self.invoke_json(
                retries=3,
                query=query,
                topic=topic,
                subject=subject,
                source_route=json.dumps(source_route, ensure_ascii=False),
            )
        except Exception as exc:
            return {
                "legal_issues": [query],
                "must_have_source_types": [],
                "reason": f"Issue analysis failed; fallback to raw query. Error: {exc}",
            }

        return normalize_issue_analysis(data, query)
