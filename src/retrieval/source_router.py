import math
import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

os.environ["USE_TF"] = "0"

load_dotenv()


SOURCE_PROFILES = [
    {
        "source_policy": "law_first",
        "description": """
        Truy vấn chỉ cần giải thích pháp luật thành văn hoặc tra cứu quy định trong pháp điển.
        Câu hỏi thường yêu cầu trả lời trực tiếp theo luật: là gì, điều kiện gì,
        thủ tục thế nào, cần giấy tờ gì, hồ sơ gồm gì, xử lý thế nào theo luật,
        quyền và nghĩa vụ ra sao, căn cứ pháp lý nào, điều luật nào áp dụng.
        Ví dụ loại truy vấn: điều kiện kết hôn là gì; thủ tục đăng ký kết hôn
        cần giấy tờ gì; ly hôn đơn phương cần điều kiện gì; hợp đồng đặt cọc
        vô hiệu xử lý thế nào; điều kiện chuyển nhượng quyền sử dụng đất là gì.
        Nguồn chính phù hợp là PHAPDIEN. ANLE chỉ là nguồn phụ nếu cần ví dụ thực tiễn.
        """,
    },
    {
        "source_policy": "case_first",
        "description": """
        Truy vấn chủ yếu muốn tìm tài liệu xét xử hoặc vụ việc cụ thể:
        bản án, án lệ, quyết định của tòa án, tranh chấp đã được giải quyết,
        nhận định của tòa, phúc thẩm, sơ thẩm, giám đốc thẩm, vụ án tương tự.
        Ví dụ loại truy vấn: có bản án nào về hợp đồng đặt cọc; tìm án lệ về
        tranh chấp đất đai; vụ án tương tự đã xét xử; nhận định của tòa trong
        bản án phúc thẩm hoặc giám đốc thẩm.
        Nguồn chính phù hợp là ANLE. PHAPDIEN chỉ là nguồn phụ để giải thích quy định nền.
        """,
    },
    {
        "source_policy": "balanced",
        "description": """
        Truy vấn nói rõ rằng cần cả hai loại nguồn trong cùng câu hỏi:
        vừa cần quy định pháp luật, điều luật, căn cứ pháp lý,
        vừa cần thực tiễn xét xử, bản án, án lệ, quyết định của tòa,
        hoặc muốn so sánh quy định với cách tòa án áp dụng trong thực tế.
        Ví dụ loại truy vấn: quy định và thực tiễn xét xử về hợp đồng đặt cọc;
        điều luật và án lệ liên quan; pháp luật quy định thế nào và tòa án
        thường xử lý ra sao.
        Cần tìm đồng thời PHAPDIEN và ANLE với tỷ lệ cân bằng.
        """,
    },
]


@dataclass
class SourceRoute:
    source_policy: str
    phapdien_k: int
    anle_k: int
    confidence: float
    matches: list
    reason: str


def get_embeddings():
    return NVIDIAEmbeddings(
        model=os.getenv("NVIDIA_EMBED_MODEL"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="END",
        max_batch_size=32,
    )


def cosine_similarity(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


@lru_cache(maxsize=1)
def get_profile_embeddings():
    embeddings = get_embeddings()
    texts = [profile["description"].strip() for profile in SOURCE_PROFILES]
    return embeddings.embed_documents(texts)


def rank_source_profiles(query):
    embeddings = get_embeddings()
    query_embedding = embeddings.embed_query(query)
    profile_embeddings = get_profile_embeddings()

    matches = []
    for profile, profile_embedding in zip(SOURCE_PROFILES, profile_embeddings):
        matches.append(
            {
                "source_policy": profile["source_policy"],
                "score": cosine_similarity(query_embedding, profile_embedding),
            }
        )

    return sorted(matches, key=lambda item: item["score"], reverse=True)


def quotas_for_policy(source_policy, total_k):
    if source_policy == "law_first":
        anle_k = min(2, max(total_k - 1, 0))
        return total_k - anle_k, anle_k

    if source_policy == "case_first":
        phapdien_k = min(2, max(total_k - 1, 0))
        return phapdien_k, total_k - phapdien_k

    phapdien_k = total_k // 2
    return phapdien_k, total_k - phapdien_k


def route_sources(query: str, total_k: int = 8) -> SourceRoute:
    matches = rank_source_profiles(query)
    top_match = matches[0] if matches else {"source_policy": "balanced", "score": 0.0}
    source_policy = top_match["source_policy"]
    phapdien_k, anle_k = quotas_for_policy(source_policy, total_k)

    return SourceRoute(
        source_policy=source_policy,
        phapdien_k=phapdien_k,
        anle_k=anle_k,
        confidence=top_match["score"],
        matches=matches,
        reason="Selected by semantic similarity to source profiles.",
    )
