from __future__ import annotations


def format_memory(memory: dict) -> str:
    lines = []

    active_context = memory.get("active_context") or {}
    if active_context:
        lines.append("Ngữ cảnh pháp lý hiện tại:")

        if active_context.get("topic_title"):
            lines.append(f"- Chủ đề: {active_context['topic_title']}")

        if active_context.get("subject_title"):
            lines.append(f"- Đề mục: {active_context['subject_title']}")

        recent_titles = active_context.get("recent_titles") or []
        if recent_titles:
            lines.append("- Nguồn gần đây:")
            for title in recent_titles[:6]:
                lines.append(f"  + {title}")

    messages = memory.get("messages") or []
    if messages:
        lines.append("")
        lines.append("Hội thoại gần đây:")
        for msg in messages[-6:]:
            lines.append(f"- {msg['role']}: {msg['content']}")

    return "\n".join(lines).strip()


def evidence_for_chat_prompt(search_result: dict) -> list:
    evidence = search_result.get("evidence") or []
    source_route = search_result.get("source_route") or {}
    source_policy = source_route.get("source_policy") or search_result.get("intent")

    if source_policy == "law_first":
        phapdien = [item for item in evidence if item.get("source_type") == "phapdien"]
        return phapdien or evidence

    return evidence
