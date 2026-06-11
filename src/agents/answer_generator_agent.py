from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.context import evidence_for_chat_prompt, format_memory


def format_evidence(evidence, max_chars: int = 2200):
    blocks = []

    for i, item in enumerate(evidence, start=1):
        source_type = item.get("source_type", "")
        title = item.get("title", "")
        url = item.get("source_url", "")
        content = item.get("expanded_content") or item.get("content", "")

        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        blocks.append(
            f"""[{i}] {source_type.upper()} - {title}
Nguồn: {url}
Nội dung:
{content}
"""
        )

    return "\n".join(blocks)


def build_answer_policy(search_result):
    source_route = search_result.get("source_route") or {}
    source_policy = source_route.get("source_policy", "balanced")
    answer_mode = search_result.get("answer_mode") or {}
    mode = answer_mode.get("mode", "normal")
    mode_policy = []

    if mode == "summary":
        mode_policy.append(
            "- The user asks for a summary. Summarize only supported key points; do not reproduce long full-text provisions or full judgments."
        )
    elif mode == "detail_case":
        mode_policy.append(
            "- The user asks for detailed case-law analysis. Prioritize ANLE/court evidence and explain facts, legal issues, reasoning, outcome, and relevance only when CONTEXT supports them."
        )
    elif mode == "full_provision":
        mode_policy.append(
            "- The user asks for exact/full provision text. If a direct exact answer is unavailable, quote the relevant provision from CONTEXT as fully as possible and do not summarize it."
        )
    elif mode == "full_case":
        mode_policy.append(
            "- The user asks for full case/precedent/judgment text. If a direct full-document answer is unavailable, quote the ANLE/court source from CONTEXT as fully as possible and do not summarize it."
        )

    if source_policy == "law_first":
        source_policy_text = """
- The question prioritizes legal rules. Focus the answer on PHAPDIEN/legal-rule sources.
- Mention ANLE/court sources only when they directly add useful case-law context.
- Do not say there is no case-law information if the user did not ask for case law.
""".strip()
        return "\n".join(mode_policy + [source_policy_text]).strip()

    if source_policy == "case_first":
        source_policy_text = """
- The question prioritizes judgments, precedents, or court practice. Focus the answer on ANLE/court sources.
- If the user is searching for cases, list the most relevant ANLE/court sources in CONTEXT, preferably 3-6 judgments/decisions when available.
- For each listed item, state the case/judgment/decision title or number, the directly related legal issue, and the supporting citation.
- Use PHAPDIEN/legal-rule sources only for necessary legal background.
- Do not turn a case-search answer into a long general legal explanation when the user only asks to find cases.
""".strip()
        return "\n".join(mode_policy + [source_policy_text]).strip()

    source_policy_text = """
- The question may need both legal rules and court practice. Separate legal-rule analysis and case-law/court-practice analysis when both are directly supported by CONTEXT.
""".strip()
    return "\n".join(mode_policy + [source_policy_text]).strip()


def evidence_context_limit(search_result):
    answer_mode = search_result.get("answer_mode") or {}
    mode = answer_mode.get("mode")

    if mode == "detail_case":
        return 8000

    if mode in {"full_provision", "full_case"}:
        return 12000

    return 2200


class AnswerGeneratorAgent(BaseAgent):
    prompt_id = "answer_generator"

    def build_prompt(
        self,
        original_query: str,
        rewrite_result: dict,
        memory: dict,
        search_result: dict,
    ) -> str:
        memory_text = format_memory(memory)
        context = format_evidence(
            evidence_for_chat_prompt(search_result),
            max_chars=evidence_context_limit(search_result),
        )
        answer_policy = build_answer_policy(search_result)
        rewritten_query = rewrite_result.get("rewritten_query") or original_query

        return self.render_prompt(
            answer_policy=answer_policy,
            original_query=original_query,
            rewritten_query=rewritten_query,
            memory=memory_text,
            context=context,
        )

    def run(
        self,
        original_query: str,
        rewrite_result: dict,
        memory: dict,
        search_result: dict,
    ) -> dict:
        memory_text = format_memory(memory)
        context = format_evidence(
            evidence_for_chat_prompt(search_result),
            max_chars=evidence_context_limit(search_result),
        )
        answer_policy = build_answer_policy(search_result)
        rewritten_query = rewrite_result.get("rewritten_query") or original_query

        answer = self.invoke_text(
            retries=5,
            answer_policy=answer_policy,
            original_query=original_query,
            rewritten_query=rewritten_query,
            memory=memory_text,
            context=context,
        )

        return {"answer": answer}
