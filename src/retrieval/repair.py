import re
import sqlite3

from src.retrieval.retriever import (
    deduplicate_evidence,
    expand_anle_evidence,
    expand_phapdien_evidence,
    search_anle,
    search_phapdien,
)


REFERENCE_DB = "data/index/phapdien_reference_index.sqlite"
TOKEN_STOPWORDS = {
    "cua",
    "va",
    "ve",
    "la",
    "co",
    "cac",
    "cho",
    "khi",
    "neu",
    "thi",
    "the",
    "nao",
    "mot",
    "nhung",
    "duoc",
    "trong",
    "voi",
    "phai",
    "can",
    "quy",
    "dinh",
}


def normalize_text(value) -> str:
    return " ".join(str(value or "").lower().split())


def text_tokens(value) -> set:
    text = normalize_text(value)
    text = re.sub(r"[^\wÀ-ỹ\s]", " ", text)
    tokens = {
        token
        for token in text.split()
        if len(token) >= 2 and token not in TOKEN_STOPWORDS
    }
    return tokens


def search_missing_hint(hint: str, source_type: str, route=None, k: int = 5) -> list:
    if source_type == "anle":
        results = search_anle(hint, k=k, route=route)
        return [
            expand_anle_evidence(item)
            for item in results
        ]

    results = search_phapdien(hint, k=k, route=route)
    return [
        expand_phapdien_evidence(item)
        for item in results
    ]


def issue_is_covered(issue: str, covered_issues: set) -> bool:
    issue_key = normalize_text(issue)

    if not issue_key:
        return True

    if issue_key in covered_issues:
        return True

    issue_tokens = set(issue_key.split())
    if not issue_tokens:
        return True

    for covered in covered_issues:
        covered_tokens = set(covered.split())
        if not covered_tokens:
            continue

        overlap = len(issue_tokens & covered_tokens) / max(len(issue_tokens), 1)
        if overlap >= 0.75:
            return True

    return False


def uncovered_issue_hints(validation: dict) -> list:
    issue_analysis = validation.get("issue_analysis") or {}
    legal_issues = issue_analysis.get("legal_issues") or []

    covered_issues = set()
    for item in validation.get("accepted") or []:
        for issue in item.get("covered_issues") or []:
            issue_key = normalize_text(issue)
            if issue_key:
                covered_issues.add(issue_key)

    output = []
    seen = set()

    for issue in legal_issues:
        issue = str(issue or "").strip()
        issue_key = normalize_text(issue)

        if not issue or issue_key in seen:
            continue

        if issue_is_covered(issue, covered_issues):
            continue

        seen.add(issue_key)
        output.append({
            "source_type": "phapdien",
            "hint": issue,
        })

    return output


def repair_query_text(query: str, validation: dict) -> str:
    parts = [query]

    for item in validation.get("missing") or []:
        if item.get("hint"):
            parts.append(str(item["hint"]))

    issue_analysis = validation.get("issue_analysis") or {}
    for issue in issue_analysis.get("legal_issues") or []:
        parts.append(str(issue))

    return " ".join(parts)


def extract_article_reference_numbers(text: str, max_range_size: int = 20) -> list:
    if not text:
        return []

    numbers = []

    range_pattern = re.compile(
        r"(?:từ\s+)?Điều\s+(\d+)\s+(?:đến|tới|-)\s+Điều\s+(\d+)",
        flags=re.IGNORECASE,
    )

    for match in range_pattern.finditer(text):
        start = int(match.group(1))
        end = int(match.group(2))

        if start > end:
            start, end = end, start

        if end - start > max_range_size:
            continue

        numbers.extend(str(number) for number in range(start, end + 1))

    numbers.extend(re.findall(r"Điều\s+(\d+)", text, flags=re.IGNORECASE))

    output = []
    seen = set()

    for number in numbers:
        if number in seen:
            continue

        seen.add(number)
        output.append(number)

    return output


def article_row_to_evidence(row) -> dict:
    article = dict(row)
    content = article.get("content_text") or ""

    return {
        "source_type": "phapdien",
        "title": article.get("article_title"),
        "content": content,
        "expanded_content": content,
        "source_url": article.get("source_url"),
        "score": 0,
        "retrieval_mode": "reference_repair",
        "context_mode": "full_article",
        "metadata": article,
    }


def score_reference_article(row, repair_text: str) -> float:
    query_tokens = text_tokens(repair_text)
    title_tokens = text_tokens(row["article_title"])
    content_tokens = text_tokens((row["content_text"] or "")[:800])

    if not query_tokens:
        return 0

    title_overlap = len(query_tokens & title_tokens)
    content_overlap = len(query_tokens & content_tokens)

    return (title_overlap * 3) + content_overlap


def reference_repair_evidence(search_result: dict, validation: dict, query: str, max_items: int = 5) -> list:
    evidence = search_result.get("evidence") or []
    repair_text = repair_query_text(query, validation)

    candidates = []
    seen = set()

    conn = sqlite3.connect(REFERENCE_DB)
    conn.row_factory = sqlite3.Row

    try:
        for item in evidence:
            if item.get("source_type") != "phapdien":
                continue

            metadata = item.get("metadata") or {}
            subject_title = metadata.get("subject_title")
            content = item.get("expanded_content") or item.get("content") or ""

            if not subject_title or not content:
                continue

            for article_number in extract_article_reference_numbers(content):
                key = (subject_title, article_number)
                if key in seen:
                    continue

                seen.add(key)
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

                score = score_reference_article(row, repair_text)
                candidates.append((score, row))
    finally:
        conn.close()

    candidates.sort(key=lambda item: item[0], reverse=True)

    return [
        article_row_to_evidence(row)
        for score, row in candidates[:max_items]
    ]


def normalize_missing(validation: dict, max_hints: int = 3) -> list:
    output = []
    seen = set()

    candidates = list(validation.get("missing") or [])
    candidates.extend(uncovered_issue_hints(validation))

    for item in candidates:
        hint = str(item.get("hint") or "").strip()
        source_type = str(item.get("source_type") or "phapdien").strip().lower()

        if not hint:
            continue

        if source_type not in {"phapdien", "anle"}:
            source_type = "phapdien"

        key = (source_type, hint.lower())
        if key in seen:
            continue

        seen.add(key)
        output.append({
            "source_type": source_type,
            "hint": hint,
        })

        if len(output) >= max_hints:
            break

    return output


def merge_repaired_evidence(original_evidence: list, repaired_evidence: list) -> list:
    merged = list(original_evidence) + list(repaired_evidence)
    return deduplicate_evidence(merged)


def repair_missing_evidence(
    query: str,
    search_result: dict,
    validation: dict,
    k_per_hint: int = 5,
    max_hints: int = 3,
) -> dict:
    missing = normalize_missing(validation, max_hints=max_hints)

    if not missing:
        return {
            "repaired": False,
            "missing": [],
            "added_evidence": [],
            "search_result": search_result,
        }

    route = search_result.get("route")
    added_evidence = reference_repair_evidence(
        search_result=search_result,
        validation=validation,
        query=query,
    )

    for item in missing:
        results = search_missing_hint(
            hint=item["hint"],
            source_type=item["source_type"],
            route=route,
            k=k_per_hint,
        )
        added_evidence.extend(results)

    merged_evidence = merge_repaired_evidence(
        search_result.get("evidence") or [],
        added_evidence,
    )

    repaired_result = dict(search_result)
    repaired_result["evidence"] = merged_evidence
    repaired_result["locked_evidence"] = search_result.get("evidence") or []
    repaired_result["repair"] = {
        "repaired": True,
        "missing": missing,
        "added_count": len(added_evidence),
        "added_titles": [
            item.get("title")
            for item in added_evidence
            if item.get("title")
        ],
    }

    return {
        "repaired": True,
        "missing": missing,
        "added_evidence": added_evidence,
        "search_result": repaired_result,
    }
