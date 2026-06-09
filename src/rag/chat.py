import json
import os
import re
import sqlite3

os.environ["USE_TF"] = "0"

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from src.memory.conversation import load_memory, save_turn
from src.rag.answer import build_answer_policy, format_evidence, invoke_with_retry
from src.retrieval.retriever import search_all
from src.retrieval.validator import apply_validation, validate_evidence
from src.retrieval.repair import repair_missing_evidence


load_dotenv()

REFERENCE_DB = "data/index/phapdien_reference_index.sqlite"


def get_llm(max_tokens=1200):
    return ChatNVIDIA(
        model=os.getenv("NVIDIA_LLM_MODEL"),
        api_key=os.getenv("NVIDIA_API_LLM") or os.getenv("NVIDIA_API_KEY"),
        temperature=0,
        max_tokens=max_tokens,
    )


def parse_json_object(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


def format_memory(memory: dict) -> str:
    lines = []

    active_context = memory.get("active_context") or {}
    if active_context:
        lines.append("Ngu canh phap ly hien tai:")

        if active_context.get("topic_title"):
            lines.append(f"- Chu de: {active_context['topic_title']}")

        if active_context.get("subject_title"):
            lines.append(f"- De muc: {active_context['subject_title']}")

        recent_titles = active_context.get("recent_titles") or []
        if recent_titles:
            lines.append("- Nguon gan day:")
            for title in recent_titles[:6]:
                lines.append(f"  + {title}")

    messages = memory.get("messages") or []
    if messages:
        lines.append("")
        lines.append("Hoi thoai gan day:")
        for msg in messages[-6:]:
            lines.append(f"- {msg['role']}: {msg['content']}")

    return "\n".join(lines).strip()


def rewrite_query_with_memory(message: str, memory: dict) -> dict:
    memory_text = format_memory(memory)

    if not memory_text:
        return {
            "needs_memory": False,
            "need_memory": False,
            "rewritten_query": message,
            "reason": "No memory available.",
        }

    prompt = f"""
Ban la bo rewrite truy van cho he thong RAG phap luat Viet Nam.

Nhiem vu:
- Chi rewrite cau hoi neu cau hoi phu thuoc vao ngu canh hoi thoai truoc.
- Khong tra loi cau hoi.
- Khong tu them dieu luat, ban an, an le ngoai MEMORY.
- Neu cau hoi da du ro, giu nguyen.
- Chi tra ve JSON hop le, dung double quotes.

MEMORY:
{memory_text}

CAU HOI HIEN TAI:
{message}

JSON format:
{{
  "needs_memory": true,
  "rewritten_query": "cau hoi da duoc viet lai",
  "reason": "ly do ngan"
}}
""".strip()

    llm = get_llm(max_tokens=500)

    try:
        response = invoke_with_retry(llm, prompt, retries=3)
        data = parse_json_object(response.content)
    except Exception:
        return {
            "needs_memory": False,
            "need_memory": False,
            "rewritten_query": message,
            "reason": "Rewrite failed, fallback to original query.",
        }

    rewritten_query = str(data.get("rewritten_query") or message).strip() or message
    needs_memory = bool(data.get("needs_memory", data.get("need_memory", False)))

    return {
        "needs_memory": needs_memory,
        "need_memory": needs_memory,
        "rewritten_query": rewritten_query,
        "reason": data.get("reason") or "",
    }


def build_chat_prompt(
    original_query: str,
    rewrite_result: dict,
    memory: dict,
    search_result: dict,
) -> str:
    memory_text = format_memory(memory)
    context = format_evidence(evidence_for_chat_prompt(search_result))
    answer_policy = build_answer_policy(search_result)

    return f"""
Ban la tro ly nghien cuu phap luat Viet Nam.

Nhiem vu:
- Chi tra loi dua tren MEMORY va NGU CANH duoc cung cap.
- Khong bia dieu luat, ban an, an le hoac tu them thong tin ngoai ngu canh.
- Neu ngu canh khong du can cu, noi ro: "Ngu canh cung cap chua du can cu de tra loi".
- Luon trich dan nguon ngay sau thong tin bang ky hieu [1], [2]...
- Khong ket luan Toa an ap dung luat dung hay sai; chi tom tat khach quan noi dung trong nguon.
- Khong trinh bay nhu tu van phap ly chinh thuc.
- Neu cau hoi khong hoi ve ban an/an le/thuc tien xet xu, khong viet cau "khong co thong tin ve ban an/an le".

Dinh huong theo loai truy van:
{answer_policy}

CAU HOI GOC:
{original_query}

CAU HOI DUNG DE TIM KIEM:
{rewrite_result["rewritten_query"]}

MEMORY:
{memory_text}

NGU CANH:
{context}

TRA LOI:
""".strip()


def evidence_for_chat_prompt(search_result: dict) -> list:
    evidence = search_result.get("evidence") or []
    source_route = search_result.get("source_route") or {}
    source_policy = source_route.get("source_policy") or search_result.get("intent")

    if source_policy == "law_first":
        phapdien = [
            item
            for item in evidence
            if item.get("source_type") == "phapdien"
        ]
        return phapdien or evidence

    return evidence


def extract_article_numbers(text: str) -> list:
    if not text:
        return []

    numbers = re.findall(r"Điều\s+(\d+)", text, flags=re.IGNORECASE)
    output = []
    seen = set()

    for number in numbers:
        if number in seen:
            continue

        seen.add(number)
        output.append(number)

    return output


def lookup_articles_from_memory(original_query: str, rewrite_result: dict, memory: dict) -> list:
    active_context = memory.get("active_context") or {}
    subject_title = active_context.get("subject_title")

    if not subject_title:
        return []

    search_text = " ".join(
        [
            original_query or "",
            rewrite_result.get("rewritten_query") or "",
        ]
    )
    article_numbers = extract_article_numbers(search_text)

    if not article_numbers:
        return []

    conn = sqlite3.connect(REFERENCE_DB)
    conn.row_factory = sqlite3.Row

    output = []

    try:
        for article_number in article_numbers:
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

            article = dict(row)
            content = article.get("content_text") or ""

            output.append(
                {
                    "source_type": "phapdien",
                    "title": article.get("article_title"),
                    "content": content,
                    "expanded_content": content,
                    "source_url": article.get("source_url"),
                    "score": 0,
                    "retrieval_mode": "memory_reference",
                    "context_mode": "full_article",
                    "metadata": article,
                }
            )
    finally:
        conn.close()

    return output


def prepend_memory_reference_evidence(search_result: dict, references: list) -> dict:
    if not references:
        return search_result

    existing = search_result.get("evidence") or []
    seen = {
        (
            item.get("source_type"),
            item.get("source_url") or item.get("title"),
        )
        for item in references
    }

    evidence = list(references)

    for item in existing:
        key = (
            item.get("source_type"),
            item.get("source_url") or item.get("title"),
        )
        if key in seen:
            continue

        seen.add(key)
        evidence.append(item)

    updated = dict(search_result)
    updated["evidence"] = evidence
    return updated

def run_validation_pipeline(
    query: str,
    search_result: dict,
    validation_mode: str = "none",
) -> tuple[dict, dict]:
    if validation_mode not in {"none", "validate", "repair"}:
        validation_mode = "none"

    debug = {
        "validation_mode": validation_mode,
        "validation_1": None,
        "validation_2": None,
        "repair": None,
    }

    if validation_mode == "none":
        return search_result, debug

    validation_1 = validate_evidence(query, search_result, max_items=8)
    debug["validation_1"] = validation_1

    validated_result = apply_validation(search_result, validation_1)

    if validation_mode == "validate":
        return validated_result, debug

    repair = repair_missing_evidence(
        query=query,
        search_result=validated_result,
        validation=validation_1,
        k_per_hint=5,
        max_hints=5,
    )

    debug["repair"] = repair

    repaired_result = repair["search_result"]

    validation_2 = validate_evidence(query, repaired_result, max_items=12)
    debug["validation_2"] = validation_2

    final_result = apply_validation(repaired_result, validation_2)

    return final_result, debug

def prepare_chat_search(session_id: str, message: str, k: int = 8, validation_mode: str = "none") -> dict:
    memory = load_memory(session_id)
    rewrite_result = rewrite_query_with_memory(message, memory)
    rewritten_query = rewrite_result["rewritten_query"]

    search_result = search_all(rewritten_query, k=k)
    memory_references = lookup_articles_from_memory(message, rewrite_result, memory)
    search_result = prepend_memory_reference_evidence(search_result, memory_references)
    
    search_result, validation_debug = run_validation_pipeline(
        query=rewritten_query,
        search_result=search_result,
        validation_mode=validation_mode,
    )
    
    return {
        'session_id': session_id,
        'message': message,
        'memory': memory,
        'rewrite_result': rewrite_result,
        'rewritten_query': rewritten_query,
        'search_result': search_result,
        'validation_debug': validation_debug,
    }


def prepair(session_id: str, message: str, k: int = 8, validation_mode: str = "none") -> dict:
    return prepare_chat_search(
        session_id=session_id,
        message=message,
        k=k,
        validation_mode=validation_mode,
    )


def chat(session_id: str, message: str, k: int = 8, validation_mode: str = "none") -> dict:
    prepared = prepare_chat_search(
        session_id=session_id,
        message=message,
        k=k,
        validation_mode=validation_mode,
    )
    
    memory = prepared['memory']
    rewrite_result = prepared['rewrite_result']
    rewritten_query = prepared['rewritten_query']
    search_result = prepared['search_result']
    validation_debug = prepared['validation_debug']
    
    prompt = build_chat_prompt(
        original_query=message,
        rewrite_result=rewrite_result,
        memory=memory,
        search_result=search_result,
    )

    llm = get_llm(max_tokens=1500)
    response = invoke_with_retry(llm, prompt)

    save_turn(
        session_id=session_id,
        user_message=message,
        assistant_answer=response.content,
        search_result=search_result,
    )

    return {
        "answer": response.content,
        "session_id": session_id,
        "original_query": message,
        "rewritten_query": rewritten_query,
        "rewrite": rewrite_result,
        "intent": search_result["intent"],
        "quotas": search_result["quotas"],
        "route": search_result.get("route"),
        "source_route": search_result.get("source_route"),
        "query_phrases": search_result.get("query_phrases"),
        "evidence": search_result["evidence"],
        "validation_mode": validation_mode,
        "validation_1": validation_debug.get("validation_1"),
        "validation_2": validation_debug.get("validation_2"),
        "repair": validation_debug.get("repair"),
        "validation": search_result.get("validation"),
        "original_evidence": search_result.get("original_evidence"),
    }
