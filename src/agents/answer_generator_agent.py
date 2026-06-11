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
            "- Người dùng yêu cầu tóm tắt. Chỉ tóm tắt các ý chính có căn cứ; không chép nguyên văn dài toàn bộ điều khoản hoặc bản án."
        )
    elif mode == "detail_case":
        mode_policy.append(
            "- Người dùng yêu cầu phân tích chi tiết án lệ/bản án. Ưu tiên nguồn ANLE, nêu sự kiện, vấn đề pháp lý, lập luận, kết quả và ý nghĩa nếu CONTEXT hỗ trợ."
        )
    elif mode == "full_provision":
        mode_policy.append(
            "- Người dùng yêu cầu nguyên văn/toàn bộ điều khoản. Nếu chưa có direct answer, hãy trích nguyên văn từ CONTEXT đầy đủ nhất có thể và không tóm tắt."
        )
    elif mode == "full_case":
        mode_policy.append(
            "- Người dùng yêu cầu toàn văn án lệ/bản án. Nếu chưa có direct answer, hãy trích nội dung nguồn ANLE đầy đủ nhất có thể và không tóm tắt."
        )

    if source_policy == "law_first":
        source_policy_text = """
- Câu hỏi ưu tiên quy định pháp luật. Trả lời trọng tâm bằng nguồn PHAPDIEN.
- Chỉ nhắc bản án/án lệ nếu nguồn ANLE có nội dung trực tiếp bổ sung cho câu trả lời.
- Không viết câu kiểu "không có thông tin về bản án/án lệ" nếu câu hỏi không yêu cầu bản án/án lệ.
""".strip()
        return "\n".join(mode_policy + [source_policy_text]).strip()

    if source_policy == "case_first":
        source_policy_text = """
- Câu hỏi ưu tiên bản án, án lệ hoặc thực tiễn xét xử. Trả lời trọng tâm bằng nguồn ANLE.
- Nếu câu hỏi đang tìm bản án, liệt kê các nguồn ANLE phù hợp nhất trong ngữ cảnh, ưu tiên 3-6 bản án/quyết định nếu có.
- Mỗi mục cần nêu số/tên bản án hoặc quyết định, vấn đề liên quan trực tiếp, và trích dẫn nguồn.
- Chỉ dùng PHAPDIEN để nêu quy định nền nếu thật sự cần.
- Không biến câu trả lời thành phần giải thích luật dài nếu câu hỏi chỉ hỏi tìm bản án.
""".strip()
        return "\n".join(mode_policy + [source_policy_text]).strip()

    source_policy_text = """
- Câu hỏi cần cả quy định pháp luật và thực tiễn xét xử. Tách rõ hai phần nếu cả hai loại nguồn đều có căn cứ trực tiếp.
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
