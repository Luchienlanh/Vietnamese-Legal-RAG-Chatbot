from __future__ import annotations

from src.agents.base import parse_json_object
from src.agents.evidence_validator_agent import (
    EvidenceValidatorAgent,
    apply_validation,
    compact_evidence_for_validation,
    dedupe_evidence,
    evidence_excerpt,
    evidence_identity,
    normalize_validator_output,
)
from src.agents.issue_analyzer_agent import IssueAnalyzerAgent, normalize_issue_analysis
from src.agents.llm import get_llm as make_llm


def get_llm(max_tokens=900):
    return make_llm(max_tokens=max_tokens, temperature=0)


def build_issue_prompt(query: str, search_result: dict) -> str:
    return IssueAnalyzerAgent().build_prompt(query, search_result)


def analyze_query_issues(query: str, search_result: dict) -> dict:
    return IssueAnalyzerAgent().run(query, search_result)


def build_validator_prompt(
    query: str,
    search_result: dict,
    compact_evidence: list,
    issue_analysis: dict,
) -> str:
    return EvidenceValidatorAgent().build_prompt(
        query=query,
        search_result=search_result,
        compact_evidence=compact_evidence,
        issue_analysis=issue_analysis,
    )


def validate_evidence(query: str, search_result: dict, max_items=10) -> dict:
    return EvidenceValidatorAgent().run(query, search_result, max_items=max_items)


def compact_evidence_for_val(evidence, max_items=10):
    return compact_evidence_for_validation(evidence, max_items=max_items)


def build_val_prompt(query, search_result, compact_evidence):
    issue_analysis = analyze_query_issues(query, search_result)
    return build_validator_prompt(query, search_result, compact_evidence, issue_analysis)


def normalize_val_output(data, evidence_count):
    return normalize_validator_output(data, evidence_count)
