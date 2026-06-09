import os
import time

os.environ["USE_TF"] = "0"

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from src.retrieval.retriever import search_all

load_dotenv()


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
- Câu hỏi ưu tiên bản án/án lệ/thực tiễn xét xử. Trả lời trọng tâm bằng nguồn ANLE.
- Nếu câu hỏi hỏi "có bản án nào" hoặc đang tìm bản án, không trả lời bằng một câu xác nhận ngắn.
- Liệt kê các nguồn ANLE phù hợp nhất trong ngữ cảnh, ưu tiên 3-6 bản án/quyết định nếu có.
- Mỗi mục cần nêu số/tên bản án hoặc quyết định, vấn đề liên quan trực tiếp, và trích dẫn nguồn.
- Chỉ liệt kê nguồn ANLE khi ngữ cảnh thể hiện trực tiếp nguồn đó liên quan tới câu hỏi; nếu liên quan mơ hồ hoặc không đủ căn cứ thì bỏ qua nguồn đó.
- Nếu ngữ cảnh của một nguồn chỉ đủ cho thấy nguồn đó liên quan tới tranh chấp/cụm tìm kiếm, nói ngắn gọn mức độ liên quan thay vì suy diễn thêm.
- Không dùng các cụm như "có thể liên quan", "có thể bao gồm", "dường như" để lấp khoảng trống.
- Không thêm đoạn "Lưu ý" hoặc kết luận tổng quát nói rằng nguồn chỉ có thể liên quan trực tiếp/gián tiếp.
- Với câu hỏi tìm bản án, kết thúc ngay sau danh sách bản án; không viết câu tổng kết, khuyến nghị hoặc nhận xét sau danh sách.
- Chỉ dùng PHAPDIEN để nêu quy định nền nếu thật sự cần.
- Không biến câu trả lời thành phần giải thích luật dài nếu câu hỏi chỉ hỏi tìm bản án.
""".strip()

    return """
- Câu hỏi cần cả quy định pháp luật và thực tiễn xét xử. Tách rõ hai phần nếu cả hai loại nguồn đều có căn cứ trực tiếp.
""".strip()


def build_prompt(query, search_result):
    context = format_evidence(search_result["evidence"])
    answer_policy = build_answer_policy(search_result)

    return f"""
Bạn là trợ lý nghiên cứu pháp luật Việt Nam.

Nhiệm vụ:
- Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.
- Không bịa điều luật, bản án, án lệ hoặc tự thêm thông tin ngoài ngữ cảnh.
- Nếu ngữ cảnh không chứa đủ thông tin để trả lời, nói rõ: "Ngữ cảnh cung cấp chưa đủ căn cứ để trả lời".
- Luôn trích dẫn nguồn ngay sau thông tin bằng ký hiệu [1], [2]...
- Không kết luận Tòa án áp dụng luật đúng hay sai; chỉ tóm tắt khách quan nội dung trong nguồn.
- Không trình bày như tư vấn pháp lý chính thức.

Định hướng theo loại truy vấn:
{answer_policy}

CÂU HỎI:
{query}

NGỮ CẢNH:
{context}

TRẢ LỜI:
""".strip()


def answer_query(query, k=8):
    search_results = search_all(query, k)
    prompt = build_prompt(query, search_results)

    llm = ChatNVIDIA(
        model=os.getenv("NVIDIA_LLM_MODEL"),
        api_key=os.getenv("NVIDIA_API_LLM") or os.getenv("NVIDIA_API_KEY"),
        temperature=0,
        max_tokens=1200,
    )

    response = invoke_with_retry(llm, prompt)

    return {
        "answer": response.content,
        "intent": search_results["intent"],
        "quotas": search_results["quotas"],
        "route": search_results.get("route"),
        "source_route": search_results.get("source_route"),
        "query_phrases": search_results.get("query_phrases"),
        "evidence": search_results["evidence"],
    }


def invoke_with_retry(llm, prompt, retries=5):
    for attempt in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            message = str(exc)
            retryable = any(code in message for code in ["429", "502", "503", "504"])
            if not retryable or attempt == retries - 1:
                raise

            wait = 10 * (attempt + 1)
            print(f"NVIDIA LLM temporary error. Retry {attempt + 1}/{retries} after {wait}s")
            time.sleep(wait)
