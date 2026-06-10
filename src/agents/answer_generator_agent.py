from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.context import evidence_for_chat_prompt, format_memory


def format_evidence(evidence):
    blocks = []

    for i, item in enumerate(evidence, start=1):
        source_type = item.get("source_type", "")
        title = item.get("title", "")
        url = item.get("source_url", "")
        content = item.get("expanded_content") or item.get("content", "")

        if len(content) > 2200:
            content = content[:2200] + "..."

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

    if source_policy == "law_first":
        return """
- Câu hỏi ưu tiên quy định pháp luật. Trả lời trọng tâm bằng nguồn PHAPDIEN.
- Chỉ nhắc bản án/án lệ nếu nguồn ANLE có nội dung trực tiếp bổ sung cho câu trả lời.
- Không viết câu kiểu "không có thông tin về bản án/án lệ" nếu câu hỏi không yêu cầu bản án/án lệ.
""".strip()

    if source_policy == "case_first":
        return """
- Câu hỏi ưu tiên bản án, án lệ hoặc thực tiễn xét xử. Trả lời trọng tâm bằng nguồn ANLE.
- Nếu câu hỏi đang tìm bản án, liệt kê các nguồn ANLE phù hợp nhất trong ngữ cảnh, ưu tiên 3-6 bản án/quyết định nếu có.
- Mỗi mục cần nêu số/tên bản án hoặc quyết định, vấn đề liên quan trực tiếp, và trích dẫn nguồn.
- Chỉ dùng PHAPDIEN để nêu quy định nền nếu thật sự cần.
- Không biến câu trả lời thành phần giải thích luật dài nếu câu hỏi chỉ hỏi tìm bản án.
""".strip()

    return """
- Câu hỏi cần cả quy định pháp luật và thực tiễn xét xử. Tách rõ hai phần nếu cả hai loại nguồn đều có căn cứ trực tiếp.
""".strip()


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
        context = format_evidence(evidence_for_chat_prompt(search_result))
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
        context = format_evidence(evidence_for_chat_prompt(search_result))
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
