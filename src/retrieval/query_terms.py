import json
import os
import re
import time

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

os.environ["USE_TF"] = "0"

load_dotenv()


def get_llm():
    return ChatNVIDIA(
        model=os.getenv("NVIDIA_LLM_MODEL"),
        api_key=os.getenv("NVIDIA_API_LLM") or os.getenv("NVIDIA_API_KEY"),
        temperature=0,
        max_tokens=300,
    )


def build_phrase_prompt(query, route=None):
    topic = ((route or {}).get("topic") or {}).get("topic_title", "")
    subject = ((route or {}).get("subject") or {}).get("subject_title", "")

    return f"""
Bạn là bộ phân tích truy vấn cho hệ thống tìm kiếm pháp luật Việt Nam.

Nhiệm vụ duy nhất: trích xuất các cụm pháp lý có nghĩa để đưa vào keyword search.

Quy tắc:
- Chỉ trả về JSON hợp lệ, không giải thích.
- Không trả lời câu hỏi pháp luật.
- Không thêm điều luật, số văn bản, án lệ hoặc nội dung không có căn cứ từ câu hỏi.
- Ưu tiên cụm danh từ pháp lý ngắn, đúng nghĩa.
- Không tách thành từng từ rời.
- Không tạo n-gram máy móc.
- Trả từ 1 đến 8 cụm.

Ngữ cảnh ontology nếu có:
- Chủ đề: {topic}
- Đề mục: {subject}

Ví dụ:
Query: Hợp đồng đặt cọc vô hiệu xử lý thế nào?
JSON: {{"phrases": ["hợp đồng đặt cọc", "đặt cọc", "hợp đồng vô hiệu", "vô hiệu"]}}

Query: Thủ tục đăng ký kết hôn cần giấy tờ gì?
JSON: {{"phrases": ["thủ tục đăng ký kết hôn", "đăng ký kết hôn", "giấy tờ đăng ký kết hôn"]}}

Query: Điều kiện chuyển nhượng quyền sử dụng đất là gì?
JSON: {{"phrases": ["điều kiện chuyển nhượng quyền sử dụng đất", "chuyển nhượng quyền sử dụng đất", "quyền sử dụng đất"]}}

Query cần phân tích:
{query}

JSON:
""".strip()


def parse_json_object(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


def clean_phrase(phrase):
    phrase = str(phrase).lower()
    phrase = re.sub(r"[^\wÀ-ỹ\s]", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase


def normalize_phrases(phrases, max_phrases=8):
    if isinstance(phrases, str):
        phrases = [phrases]

    output = []
    seen = set()

    for phrase in phrases:
        phrase = clean_phrase(phrase)
        if not phrase:
            continue

        if len(phrase.split()) > 10:
            continue

        if phrase in seen:
            continue

        seen.add(phrase)
        output.append(phrase)

        if len(output) >= max_phrases:
            break

    return output


def extract_legal_phrases(query, route=None, retries=3):
    prompt = build_phrase_prompt(query, route=route)
    llm = get_llm()

    for attempt in range(retries):
        try:
            response = llm.invoke(prompt)
            data = parse_json_object(response.content)
            phrases = data.get("phrases", [])
            if not isinstance(phrases, list):
                return []
            return normalize_phrases(phrases)
        except Exception as exc:
            message = str(exc)
            retryable = any(code in message for code in ["429", "502", "503", "504"])
            if not retryable or attempt == retries - 1:
                return []

            time.sleep(5 * (attempt + 1))

    return []


def quote_fts_phrase(phrase):
    phrase = clean_phrase(phrase)
    phrase = phrase.replace('"', " ")
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return f'"{phrase}"'


def build_fts_query(phrases):
    phrases = normalize_phrases(phrases)
    if not phrases:
        return ""

    return " OR ".join(quote_fts_phrase(phrase) for phrase in phrases)
