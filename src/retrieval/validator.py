import json
import os
import re

os.environ["USE_TF"] = "0"

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from src.rag.answer import invoke_with_retry


load_dotenv()


def get_llm(max_tokens=900):
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


def evidence_excerpt(item, max_chars=1000):
    content = item.get("expanded_content") or item.get("content") or ""
    content = re.sub(r"\s+", " ", str(content)).strip()

    if len(content) > max_chars:
        return content[:max_chars] + "..."

    return content


def compact_evidence_for_validation(evidence, max_items=10):
    output = []

    for i, item in enumerate(evidence[:max_items], start=1):
        metadata = item.get("metadata") or {}

        output.append(
            {
                "index": i,
                "source_type": item.get("source_type"),
                "title": item.get("title"),
                "context_mode": item.get("context_mode"),
                "source_url": item.get("source_url"),
                "metadata": {
                    "topic_title": metadata.get("topic_title"),
                    "subject_title": metadata.get("subject_title"),
                    "article_title": metadata.get("article_title"),
                    "original_article_number": metadata.get("original_article_number"),
                    "doc_code": metadata.get("doc_code"),
                    "subject": metadata.get("subject"),
                },
                "excerpt": evidence_excerpt(item),
            }
        )

    return output


def build_issue_prompt(query: str, search_result: dict) -> str:
    route = search_result.get("route") or {}
    topic = (route.get("topic") or {}).get("topic_title")
    subject = (route.get("subject") or {}).get("subject_title")
    source_route = search_result.get("source_route") or {}

    return f"""
Ban la bo phan tich truy van cho he thong RAG phap luat Viet Nam.

Nhiem vu:
- Phan tich QUERY thanh cac legal issues/concepts can co de tra loi dung.
- Khong tra loi cau hoi.
- Khong them dieu luat, ban an, an le, so dieu cu the neu QUERY khong neu.
- Moi issue phai la cum phap ly ngan, tong quat, co y nghia.
- Neu QUERY co nhieu y phap ly, tach thanh nhieu issue.
- Neu QUERY co cum phap ly phuc hop, tach theo cac lop nghia:
  che dinh/doi tuong phap ly, tinh trang/dieu kien phap ly, va hau qua/cach xu ly.
- Khong lap lai toan bo QUERY nhu mot issue duy nhat neu QUERY chua nhieu concept.
- Chi tra JSON hop le, dung double quotes.

Ontology context neu co:
topic={topic}
subject={subject}
source_route={source_route}

QUERY:
{query}

JSON format:
{{
  "legal_issues": [
    "issue 1",
    "issue 2"
  ],
  "must_have_source_types": ["phapdien"],
  "reason": "ly do ngan"
}}
""".strip()


def normalize_issue_analysis(data: dict, query: str) -> dict:
    issues = data.get("legal_issues") or []
    normalized_issues = []
    seen = set()

    for issue in issues:
        issue = str(issue).strip()
        if not issue:
            continue

        key = issue.lower()
        if key in seen:
            continue

        seen.add(key)
        normalized_issues.append(issue)

    if not normalized_issues:
        normalized_issues = [query]

    source_types = data.get("must_have_source_types") or []
    normalized_source_types = []

    for source_type in source_types:
        source_type = str(source_type).strip().lower()
        if source_type in {"phapdien", "anle"}:
            normalized_source_types.append(source_type)

    return {
        "legal_issues": normalized_issues[:8],
        "must_have_source_types": normalized_source_types[:3],
        "reason": str(data.get("reason") or ""),
    }


def analyze_query_issues(query: str, search_result: dict) -> dict:
    prompt = build_issue_prompt(query, search_result)
    llm = get_llm(max_tokens=500)

    try:
        response = invoke_with_retry(llm, prompt, retries=3)
        data = parse_json_object(response.content)
    except Exception as exc:
        return {
            "legal_issues": [query],
            "must_have_source_types": [],
            "reason": f"Issue analysis failed; fallback to raw query. Error: {exc}",
        }

    return normalize_issue_analysis(data, query)


def build_validator_prompt(
    query: str,
    search_result: dict,
    compact_evidence: list,
    issue_analysis: dict,
) -> str:
    intent = search_result.get("intent")
    quotas = search_result.get("quotas")
    source_route = search_result.get("source_route") or {}

    return f"""
Ban la evidence validator cho he thong RAG phap luat Viet Nam.

Nhiem vu:
- Doc QUERY, LEGAL_ISSUES va danh sach EVIDENCE.
- Chon evidence thuc su giup tra loi QUERY.
- Evidence duoc accept neu no truc tiep cover it nhat mot legal issue quan trong.
- Evidence ve che dinh nen duoc accept neu query can hieu che dinh do de tra loi dung,
  ngay ca khi evidence do khong cover tinh trang/hau qua phap ly khac trong query.
- Neu title/excerpt cua evidence dinh nghia hoac quy dinh truc tiep mot che dinh phap ly
  duoc neu ro trong QUERY, phai accept evidence do nhu "foundational_context".
- Khong reject foundational_context chi vi no khong cover phan dieu kien, vo hieu,
  hau qua, trach nhiem, thu tuc hoac thuc tien xet xu.
- Khong yeu cau mot evidence phai cover tat ca legal issues.
- Reject evidence neu chi trung keyword nhung khong cover legal issue nao.
- Reject evidence sai linh vuc, qua rong, qua gian tiep, hoac de gay nhieu khi answer.
- Chi dua missing khi mot legal issue quan trong chua co evidence nao cover.
- Khong tao missing ve thoi hieu, thu tuc, tham quyen neu QUERY khong hoi nhung noi dung do.
- Khong tra loi cau hoi.
- Khong tu them noi dung phap luat ngoai evidence.
- Chi tra JSON hop le, dung double quotes.

Source preference:
- Neu query hoi quy dinh phap luat, uu tien PHAPDIEN truc tiep.
- Neu query hoi ban an/an le/thuc tien xet xu, uu tien ANLE truc tiep.
- Neu query hoi ca quy dinh va thuc tien, giu ca PHAPDIEN va ANLE neu deu cover legal issues.

QUERY:
{query}

ROUTE:
intent={intent}
quotas={quotas}
source_route={source_route}

LEGAL_ISSUES:
{json.dumps(issue_analysis, ensure_ascii=False, indent=2)}

EVIDENCE:
{json.dumps(compact_evidence, ensure_ascii=False, indent=2)}

JSON format:
{{
  "accepted": [
    {{
      "index": 1,
      "covered_issues": ["issue"],
      "reason": "ly do ngan"
    }}
  ],
  "rejected": [
    {{
      "index": 2,
      "reason": "ly do ngan"
    }}
  ],
  "missing": [
    {{
      "source_type": "phapdien",
      "hint": "legal issue con thieu neu co"
    }}
  ]
}}
""".strip()


def normalize_validator_output(data: dict, evidence_count: int) -> dict:
    accepted = []
    rejected = []
    missing = []

    for item in data.get("accepted") or []:
        try:
            index = int(item.get("index"))
        except Exception:
            continue

        if not 1 <= index <= evidence_count:
            continue

        covered_issues = item.get("covered_issues") or []
        covered_issues = [
            str(issue).strip()
            for issue in covered_issues
            if str(issue).strip()
        ]

        accepted.append(
            {
                "index": index,
                "covered_issues": covered_issues,
                "reason": str(item.get("reason") or ""),
            }
        )

    accepted_indexes = {item["index"] for item in accepted}

    for item in data.get("rejected") or []:
        try:
            index = int(item.get("index"))
        except Exception:
            continue

        if not 1 <= index <= evidence_count:
            continue

        if index in accepted_indexes:
            continue

        rejected.append(
            {
                "index": index,
                "reason": str(item.get("reason") or ""),
            }
        )

    for item in data.get("missing") or []:
        hint = str(item.get("hint") or "").strip()
        if not hint:
            continue

        source_type = str(item.get("source_type") or "phapdien").strip().lower()
        if source_type not in {"phapdien", "anle"}:
            source_type = "phapdien"

        missing.append(
            {
                "source_type": source_type,
                "hint": hint,
            }
        )

    return {
        "accepted": accepted,
        "rejected": rejected,
        "missing": missing,
    }


def validate_evidence(query: str, search_result: dict, max_items=10) -> dict:
    evidence = search_result.get("evidence") or []
    compact = compact_evidence_for_validation(evidence, max_items=max_items)

    if not compact:
        return {
            "issue_analysis": {
                "legal_issues": [query],
                "must_have_source_types": [],
                "reason": "No evidence to validate.",
            },
            "accepted": [],
            "rejected": [],
            "missing": [],
            "curated_evidence": [],
            "raw": None,
        }

    issue_analysis = analyze_query_issues(query, search_result)
    prompt = build_validator_prompt(query, search_result, compact, issue_analysis)
    llm = get_llm(max_tokens=900)

    try:
        response = invoke_with_retry(llm, prompt)
        raw_data = parse_json_object(response.content)
        normalized = normalize_validator_output(raw_data, len(compact))
    except Exception as exc:
        fallback_evidence = evidence[:max_items]
        return {
            "issue_analysis": issue_analysis,
            "accepted": [
                {
                    "index": i,
                    "covered_issues": [],
                    "reason": "Validator failed; fallback accepted original evidence order.",
                }
                for i in range(1, len(fallback_evidence) + 1)
            ],
            "rejected": [],
            "missing": [],
            "curated_evidence": fallback_evidence,
            "raw": {"error": str(exc)},
        }

    accepted_indexes = [item["index"] for item in normalized["accepted"]]
    curated_evidence = [
        evidence[index - 1]
        for index in accepted_indexes
        if 1 <= index <= len(evidence)
    ]

    if not curated_evidence:
        curated_evidence = evidence[:min(4, len(evidence))]

    return {
        "issue_analysis": issue_analysis,
        **normalized,
        "curated_evidence": curated_evidence,
        "raw": raw_data,
    }


def evidence_identity(item: dict):
    metadata = item.get("metadata") or {}

    return (
        item.get("source_type"),
        metadata.get("parent_uid")
        or metadata.get("article_anchor")
        or metadata.get("doc_name")
        or item.get("source_url")
        or item.get("title"),
    )


def dedupe_evidence(evidence: list) -> list:
    output = []
    seen = set()

    for item in evidence:
        key = evidence_identity(item)
        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


def apply_validation(search_result: dict, validation: dict) -> dict:
    updated = dict(search_result)
    locked_evidence = search_result.get("locked_evidence") or []
    curated_evidence = validation.get("curated_evidence") or []

    updated["original_evidence"] = search_result.get("evidence") or []
    updated["evidence"] = dedupe_evidence(locked_evidence + curated_evidence)
    updated["validation"] = validation
    return updated


def compact_evidence_for_val(evidence, max_items=10):
    return compact_evidence_for_validation(evidence, max_items=max_items)


def build_val_prompt(query, search_result, compact_evidence):
    issue_analysis = analyze_query_issues(query, search_result)
    return build_validator_prompt(query, search_result, compact_evidence, issue_analysis)


def normalize_val_output(data, evidence_count):
    return normalize_validator_output(data, evidence_count)
