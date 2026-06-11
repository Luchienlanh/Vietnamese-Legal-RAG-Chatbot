from __future__ import annotations

import re
import unicodedata

from src.retrieval.retriever import expand_anle_evidence, expand_anle_full_document


def normalize_vietnamese(text: str | None) -> str:
    if not text:
        return ""

    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )

    return re.sub(r"\s+", " ", without_marks.replace("đ", "d").replace("Đ", "D")).lower().strip()


def source_title(item: dict) -> str:
    metadata = item.get("metadata") or {}

    return (
        item.get("title")
        or metadata.get("article_title")
        or metadata.get("title")
        or metadata.get("subject")
        or metadata.get("doc_name")
        or "Nguồn pháp lý"
    )


def source_url(item: dict) -> str:
    metadata = item.get("metadata") or {}

    return (
        item.get("source_url")
        or metadata.get("source_url")
        or metadata.get("detail_url")
        or metadata.get("pdf_url")
        or ""
    )


def source_content(item: dict) -> str:
    return (item.get("expanded_content") or item.get("content") or "").strip()


def matches_article(item: dict, article_numbers: list[str]) -> bool:
    if not article_numbers:
        return True

    metadata = item.get("metadata") or {}
    haystack = normalize_vietnamese(
        " ".join(
            str(value)
            for value in [
                item.get("title"),
                metadata.get("article_title"),
                metadata.get("original_article_number"),
                metadata.get("source_note_text"),
            ]
            if value
        )
    )

    return any(
        article == normalize_vietnamese(metadata.get("original_article_number"))
        or f"dieu {article}" in haystack
        or f".{article}." in haystack
        for article in article_numbers
    )


def select_phapdien_item(evidence: list[dict], answer_mode: dict) -> dict | None:
    phapdien = [
        item
        for item in evidence
        if item.get("source_type") == "phapdien" and source_content(item)
    ]

    if not phapdien:
        return None

    article_numbers = answer_mode.get("article_numbers") or []

    for item in phapdien:
        if matches_article(item, article_numbers):
            return item

    return phapdien[0]


def line_starts_clause(line: str, clause_number: str) -> bool:
    normalized = normalize_vietnamese(line).lstrip()
    escaped = re.escape(str(clause_number))

    return bool(
        re.match(rf"^(?:khoan\s+)?{escaped}(?:[\).:]|\s+)", normalized)
    )


def line_starts_any_clause(line: str) -> bool:
    normalized = normalize_vietnamese(line).lstrip()
    return bool(re.match(r"^(?:khoan\s+)?[0-9]+[a-z0-9./-]*(?:[\).:]|\s+)", normalized))


def extract_clause_text(content: str, clause_number: str) -> str | None:
    lines = content.splitlines()
    start_index = None

    for index, line in enumerate(lines):
        if line_starts_clause(line, clause_number):
            start_index = index
            break

    if start_index is not None:
        end_index = len(lines)

        for index in range(start_index + 1, len(lines)):
            if line_starts_any_clause(lines[index]):
                end_index = index
                break

        extracted = "\n".join(lines[start_index:end_index]).strip()
        if extracted:
            return extracted

    normalized = normalize_vietnamese(content)
    escaped = re.escape(str(clause_number))
    match = re.search(
        rf"(?:khoan\s+)?{escaped}(?:[\).:]|\s+).+?(?=(?:khoan\s+)?[0-9]+[a-z0-9./-]*(?:[\).:]|\s+)|$)",
        normalized,
        flags=re.DOTALL,
    )

    if match:
        return match.group(0).strip()

    return None


def build_full_provision_answer(item: dict, answer_mode: dict) -> str | None:
    content = source_content(item)
    if not content:
        return None

    article_numbers = answer_mode.get("article_numbers") or []
    clause_numbers = answer_mode.get("clause_numbers") or []
    title = source_title(item)
    url = source_url(item)

    descriptor = title
    if article_numbers:
        descriptor = f"Điều {article_numbers[0]} - {title}"

    if clause_numbers:
        clause_text = extract_clause_text(content, clause_numbers[0])
        if clause_text:
            descriptor = f"Khoản {clause_numbers[0]} {descriptor}"
            content = clause_text
        else:
            content = (
                "Không tách được chính xác khoản được yêu cầu từ cấu trúc văn bản hiện có. "
                "Dưới đây là toàn bộ điều tìm được để bạn đối chiếu:\n\n"
                f"{content}"
            )

    answer = [
        f"Dưới đây là nguyên văn {descriptor} theo nguồn được truy xuất [1]:",
        "",
        content,
        "",
        f"Nguồn: {title} [1]",
    ]

    if url:
        answer.append(f"Liên kết nguồn: {url}")

    answer.append("Thông tin chỉ hỗ trợ tra cứu, không thay thế tư vấn pháp lý.")

    return "\n".join(answer).strip()


def find_memory_anle_candidates(memory: dict) -> list[dict]:
    output = []

    for item in memory.get("last_evidence") or []:
        if item.get("source_type") != "anle":
            continue

        output.append(
            {
                "source_type": "anle",
                "title": item.get("title"),
                "source_url": item.get("source_url"),
                "metadata": item.get("metadata") or {},
                "retrieval_mode": "memory_reference",
                "context_mode": item.get("context_mode"),
                "score": 0,
            }
        )

    return output


def select_anle_item(search_result: dict, answer_mode: dict, memory: dict) -> dict | None:
    candidates = []

    if answer_mode.get("uses_reference"):
        candidates.extend(find_memory_anle_candidates(memory))

    candidates.extend(
        item
        for item in search_result.get("evidence") or []
        if item.get("source_type") == "anle"
    )

    if not candidates:
        return None

    for item in candidates:
        expanded = expand_anle_full_document(item)
        if expanded.get("context_mode") == "anle_full_document":
            return expanded

    return None


def build_full_case_answer(item: dict) -> str | None:
    content = source_content(item)
    if not content:
        return None

    title = source_title(item)
    url = source_url(item)
    truncated = (item.get("metadata") or {}).get("full_document_truncated")

    answer = [
        f"Dưới đây là toàn văn nguồn án lệ/bản án được truy xuất: {title} [1].",
    ]

    if truncated:
        answer.append(
            "Lưu ý: nội dung nguồn quá dài nên hệ thống đã rút gọn theo giới hạn hiển thị hiện tại."
        )

    answer.extend(
        [
            "",
            content,
            "",
            f"Nguồn: {title} [1]",
        ]
    )

    if url:
        answer.append(f"Liên kết nguồn: {url}")

    answer.append("Thông tin chỉ hỗ trợ tra cứu, không thay thế tư vấn pháp lý.")

    return "\n".join(answer).strip()


def prepend_unique(evidence: list[dict], first_item: dict) -> list[dict]:
    key = (
        first_item.get("source_type"),
        first_item.get("source_url") or first_item.get("title"),
    )
    output = [first_item]

    for item in evidence:
        item_key = (
            item.get("source_type"),
            item.get("source_url") or item.get("title"),
        )
        if item_key == key:
            continue

        output.append(item)

    return output


def apply_answer_mode(search_result: dict, answer_mode: dict, memory: dict) -> dict:
    mode = answer_mode.get("mode")
    updated = dict(search_result)
    evidence = list(search_result.get("evidence") or [])
    updated["answer_mode"] = answer_mode

    if mode == "full_provision":
        item = select_phapdien_item(evidence, answer_mode)
        if item:
            answer = build_full_provision_answer(item, answer_mode)
            if answer:
                updated["evidence"] = prepend_unique(evidence, item)
                updated["direct_answer"] = answer

        return updated

    if mode == "full_case":
        item = select_anle_item(search_result, answer_mode, memory)
        if item:
            answer = build_full_case_answer(item)
            if answer:
                updated["evidence"] = prepend_unique(evidence, item)
                updated["direct_answer"] = answer

        return updated

    if mode == "detail_case":
        detailed = []
        for item in evidence:
            if item.get("source_type") == "anle":
                item = expand_anle_evidence(item, before=8, after=8, max_chars=14000)
                if item.get("context_mode") == "anle_paragraph_window":
                    item["context_mode"] = "anle_detail_window"

            detailed.append(item)

        updated["evidence"] = detailed

    return updated
