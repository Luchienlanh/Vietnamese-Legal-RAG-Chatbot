import os
import re
import sqlite3
os.environ["USE_TF"] = "0"

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from src.retrieval.ontology_router import route_query
from src.retrieval.query_terms import build_fts_query, extract_legal_phrases
from src.retrieval.source_router import route_sources
import pandas as pd
from pathlib import Path

load_dotenv()
_anle_documents_cache = None

REFERENCE_DB = "data/index/phapdien_reference_index.sqlite"
PHAPDIEN_FTS_DB = "data/index/phapdien_fts.sqlite"
ANLE_DOCUMENTS_PATH = "data/huggingface/anle-toaan-gov-vn/documents-00000-of-00001.parquet"
ANLE_SENTENCES_DIR = Path("data/huggingface/anle-toaan-gov-vn")

embeddings = NVIDIAEmbeddings(
    model=os.getenv("NVIDIA_EMBED_MODEL"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    truncate="END",
    max_batch_size=32,
)



def search_phapdien_fts(phrases, k=20):
    fts_query = build_fts_query(phrases)
    if not fts_query:
        return []

    conn = sqlite3.connect(PHAPDIEN_FTS_DB)
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """
            SELECT
                parent_uid,
                article_anchor,
                topic_id,
                subject_id,
                topic_title,
                subject_title,
                article_title,
                chapter_title,
                source_note_text,
                source_url,
                content_text,
                bm25(phapdien_fts) AS score
            FROM phapdien_fts
            WHERE phapdien_fts MATCH ?
            ORDER BY bm25(phapdien_fts)
            LIMIT ?
            """,
            (fts_query, k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    output = []
    for row in rows:
        metadata = dict(row)
        content = metadata.pop("content_text") or ""
        score = metadata.pop("score")

        output.append({
            "source_type": "phapdien",
            "title": metadata.get("article_title"),
            "content": content,
            "source_url": metadata.get("source_url"),
            "score": score,
            "metadata": metadata,
            "retrieval_mode": "fts",
            "matched_phrases": phrases,
        })

    return output

def evidence_key(item):
    meta = item.get("metadata", {})
    return (
        item.get("source_type"),
        meta.get("parent_uid")
        or meta.get("article_anchor")
        or meta.get("doc_name")
        or item.get("source_url")
        or item.get("title"),
    )


def rrf_merge(result_lists, k=60, weights=None):
    weights = weights or [1.0 for _ in result_lists]
    scores = {}
    items = {}
    modes = {}

    for result_list, weight in zip(result_lists, weights):
        seen_in_list = set()

        for rank, item in enumerate(result_list, start=1):
            key = evidence_key(item)

            if key in seen_in_list:
                continue

            seen_in_list.add(key)
            scores[key] = scores.get(key, 0) + weight / (k + rank)
            modes.setdefault(key, set()).add(item.get("retrieval_mode", "unknown"))

            if key not in items:
                items[key] = item

    merged = list(items.values())
    merged.sort(key=lambda item: scores[evidence_key(item)], reverse=True)

    for item in merged:
        key = evidence_key(item)
        item["rrf_score"] = scores[key]
        item["retrieval_modes"] = sorted(modes.get(key, []))
        if len(item["retrieval_modes"]) > 1:
            item["retrieval_mode"] = "hybrid"

    return merged


def enrich_phapdien_metadata(metadata):
    parent_uid = metadata.get("parent_uid")
    article_anchor = metadata.get("article_anchor")

    if not parent_uid and not article_anchor:
        return metadata

    conn = sqlite3.connect(REFERENCE_DB)
    conn.row_factory = sqlite3.Row

    row = None

    if parent_uid:
        row = conn.execute(
            "SELECT * FROM articles WHERE parent_uid = ?",
            (parent_uid,),
        ).fetchone()

    if row is None and article_anchor:
        row = conn.execute(
            "SELECT * FROM articles WHERE article_anchor = ? LIMIT 1",
            (article_anchor,),
        ).fetchone()

    conn.close()

    if row is None:
        return metadata

    enriched = dict(metadata)

    for key in [
        "topic_id",
        "subject_id",
        "source_item_id",
        "original_article_number",
        "source_instrument_key",
    ]:
        enriched[key] = row[key]

    return enriched


def build_route_context(route):
    if not route:
        return ""

    lines = []
    topic = route.get("topic") or {}
    subject = route.get("subject") or {}
    glossary_hits = route.get("glossary_hits") or []

    if topic.get("topic_title"):
        lines.append(f"Ontology topic: {topic['topic_title']}")

    if subject.get("subject_title"):
        lines.append(f"Ontology subject: {subject['subject_title']}")

    if glossary_hits:
        terms = ", ".join(hit["term"] for hit in glossary_hits if hit.get("term"))
        if terms:
            lines.append(f"Ontology legal terms: {terms}")

    return "\n".join(lines)

def get_full_phapdien_article(metadata):
    parent_uid = metadata.get("parent_uid")
    article_anchor = metadata.get("article_anchor")

    if not parent_uid and not article_anchor:
        return None

    conn = sqlite3.connect(REFERENCE_DB)
    conn.row_factory = sqlite3.Row

    row = None

    if parent_uid:
        row = conn.execute(
            """
            SELECT *
            FROM articles
            WHERE parent_uid = ?
            LIMIT 1
            """,
            (parent_uid,),
        ).fetchone()

    if row is None and article_anchor:
        row = conn.execute(
            """
            SELECT *
            FROM articles
            WHERE article_anchor = ?
            LIMIT 1
            """,
            (article_anchor,),
        ).fetchone()

    conn.close()

    if row is None:
        return None

    return dict(row)

def expand_phapdien_evidence(item):
    if item.get("source_type") != "phapdien":
        return item

    metadata = item.get("metadata", {})
    article = get_full_phapdien_article(metadata)

    if not article:
        return item

    expanded = dict(item)
    expanded_metadata = dict(metadata)

    for key in [
        "parent_uid",
        "article_anchor",
        "topic_id",
        "subject_id",
        "topic_title",
        "subject_title",
        "article_title",
        "chapter_title",
        "source_note_text",
        "source_url",
        "source_item_id",
        "original_article_number",
        "source_instrument_key",
    ]:
        expanded_metadata[key] = article.get(key)

    expanded["title"] = article.get("article_title") or item.get("title")
    expanded["source_url"] = article.get("source_url") or item.get("source_url")
    expanded["content"] = article.get("content_text") or item.get("content")
    expanded["metadata"] = expanded_metadata
    expanded["expanded_content"] = expanded["content"]
    expanded["context_mode"] = "full_article"

    return expanded

def expand_anle_evidence(item, before=3, after=3, max_chars=6000):
    if item.get("source_type") != "anle":
        return item

    metadata = item.get("metadata", {})
    doc_name = metadata.get("doc_name")
    paragraph_id = metadata.get("paragraph_id")

    if not doc_name or not paragraph_id:
        return item

    paragraphs_by_doc = load_anle_paragraphs()
    paragraphs = paragraphs_by_doc.get(doc_name)

    if not paragraphs:
        return item

    current_index = None
    for i, paragraph in enumerate(paragraphs):
        if paragraph["paragraph_id"] == paragraph_id:
            current_index = i
            break

    if current_index is None:
        return item

    start = max(0, current_index - before)
    end = min(len(paragraphs), current_index + after + 1)
    window = paragraphs[start:end]

    content_parts = []

    title = item.get("title") or metadata.get("title") or metadata.get("subject") or doc_name
    content_parts.append(f"Thông tin nguồn: {title}")

    detail_url = item.get("source_url") or metadata.get("detail_url")
    if detail_url:
        content_parts.append(f"Nguồn: {detail_url}")

    content_parts.append("Đoạn liên quan trong bản án/quyết định:")

    for paragraph in window:
        marker = paragraph.get("paragraph_marker") or paragraph["paragraph_id"]
        page = paragraph.get("page")
        text = paragraph.get("text", "")

        prefix = f"[{marker}"
        if page:
            prefix += f", trang {page}"
        prefix += "]"

        content_parts.append(f"{prefix} {text}")

    expanded_content = "\n".join(content_parts).strip()

    if len(expanded_content) > max_chars:
        expanded_content = expanded_content[:max_chars] + "..."

    expanded = dict(item)
    expanded["content"] = expanded_content
    expanded["expanded_content"] = expanded_content
    expanded["context_mode"] = "anle_paragraph_window"
    expanded["metadata"] = dict(metadata)
    expanded["metadata"]["window_before"] = before
    expanded["metadata"]["window_after"] = after

    return expanded

_anle_paragraphs_cache = None


def load_anle_paragraphs():
    global _anle_paragraphs_cache

    if _anle_paragraphs_cache is not None:
        return _anle_paragraphs_cache

    frames = []
    for path in sorted(ANLE_SENTENCES_DIR.glob("sentences-*.parquet")):
        frames.append(pd.read_parquet(path))

    df = pd.concat(frames, ignore_index=True)

    paragraphs = {}

    grouped = df.sort_values(
        ["doc_name", "global_index"]
    ).groupby(["doc_name", "paragraph_id"], sort=False)

    for (doc_name, paragraph_id), group in grouped:
        text = " ".join(group["text"].dropna().astype(str).tolist()).strip()

        if not text:
            continue

        first = group.iloc[0]

        paragraphs.setdefault(doc_name, []).append({
            "paragraph_id": paragraph_id,
            "section_id": first.get("section_id"),
            "section_kind": first.get("section_kind"),
            "paragraph_kind": first.get("paragraph_kind"),
            "paragraph_marker": first.get("paragraph_marker"),
            "page": first.get("page"),
            "global_index": int(group["global_index"].min()),
            "text": text,
        })

    for doc_name in paragraphs:
        paragraphs[doc_name].sort(key=lambda x: x["global_index"])

    _anle_paragraphs_cache = paragraphs
    return _anle_paragraphs_cache

def build_phapdien_query(query: str, route=None) -> str:
    route_context = build_route_context(route)

    return f"""
User question: {query}

{route_context}

Retrieve codified Vietnamese legal provisions by legal topic, legal subject,
article title, source instrument, and legal issue. Prefer provisions matching
the ontology topic and subject above when present.
""".strip()


def build_anle_query(query: str, route=None) -> str:
    route_context = build_route_context(route)

    return f"""
User question: {query}

{route_context}

Retrieve Vietnamese court judgments, decisions, precedents, dispute facts,
legal reasoning, and practical adjudication materials matching the ontology
topic and subject above when present.
""".strip()


def load_phapdien():
    return Chroma(
        collection_name="phapdien",
        persist_directory="data/vectorstores/phapdien_chroma_nvidia",
        embedding_function=embeddings,
    )



def load_anle_documents():
    global _anle_documents_cache

    if _anle_documents_cache is None:
        df = pd.read_parquet(ANLE_DOCUMENTS_PATH)
        _anle_documents_cache = {
            row["doc_name"]: row.to_dict()
            for _, row in df.iterrows()
        }

    return _anle_documents_cache

def load_anle():
    return Chroma(
        collection_name="anle",
        persist_directory="data/vectorstores/anle_chroma_nvidia",
        embedding_function=embeddings,
    )


def normalize_result(source_type, doc, score):
    metadata = dict(doc.metadata)

    if source_type == "phapdien":
        metadata = enrich_phapdien_metadata(metadata)
        source_url = metadata.get("source_url")
        title = metadata.get("article_title")
    else:
        source_url = metadata.get("detail_url") or metadata.get("pdf_url")
        title = metadata.get("title") or metadata.get("subject")

    return {
        "source_type": source_type,
        "title": title,
        "content": doc.page_content,
        "source_url": source_url,
        "score": score,
        'retrieval_mode': "vector",
        "metadata": metadata,
    }


def search_phapdien(query, k=50, route=None):
    vectorstore = load_phapdien()
    search_query = build_phapdien_query(query, route=route)
    results = vectorstore.similarity_search_with_score(search_query, k=k)
    return [normalize_result("phapdien", doc, score) for doc, score in results]


def search_anle(query, k=30, route=None):
    vectorstore = load_anle()
    search_query = build_anle_query(query, route=route)
    results = vectorstore.similarity_search_with_score(search_query, k=k)
    return [normalize_result("anle", doc, score) for doc, score in results]


def get_source_quotas(route, total_k: int = 8):
    source_policy = (route or {}).get("source_policy", "balanced")

    if source_policy == "law_first":
        anle_k = min(2, max(total_k - 1, 0))
        return {"phapdien": total_k - anle_k, "anle": anle_k}

    if source_policy == "case_first":
        phapdien_k = min(2, max(total_k - 1, 0))
        return {"phapdien": phapdien_k, "anle": total_k - phapdien_k}

    return {"phapdien": total_k // 2, "anle": total_k - (total_k // 2)}


def normalize_key_text(value):
    if not value:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip().lower()


def evidence_identity(item):
    meta = item.get("metadata", {})
    source_type = item.get("source_type")

    if source_type == "anle":
        return (
            source_type,
            normalize_key_text(
                meta.get("title")
                or item.get("title")
                or meta.get("doc_name")
                or item.get("source_url")
            ),
        )

    return (
        source_type,
        normalize_key_text(
            meta.get("parent_uid")
            or meta.get("article_anchor")
            or item.get("source_url")
            or item.get("title")
        ),
    )


def deduplicate_evidence(evidence):
    seen = set()
    output = []

    for item in evidence:
        key = evidence_identity(item)
        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output

def rerank_evidence(evidence, query):
    def sort_key(item):
        if "rrf_score" in item:
            return (0, -item["rrf_score"])

        return (1, item.get("score", 0))

    return sorted(evidence, key=sort_key)


def diversify_evidence(evidence, quotas):
    counts = {}
    output = []

    for item in evidence:
        source = item["source_type"]
        allowed = quotas.get(source, 0)
        current = counts.get(source, 0)

        if current >= allowed:
            continue

        output.append(item)
        counts[source] = current + 1

    return output


def order_evidence_by_policy(evidence, source_policy):
    priorities_by_policy = {
        "case_first": {"anle": 0, "phapdien": 1},
        "law_first": {"phapdien": 0, "anle": 1},
    }
    priority = priorities_by_policy.get(source_policy)

    if not priority:
        return evidence

    return sorted(
        evidence,
        key=lambda item: priority.get(item.get("source_type"), 99),
    )


def source_route_get(source_route, key, default=None):
    if isinstance(source_route, dict):
        return source_route.get(key, default)
    return getattr(source_route, key, default)


def route_for_phapdien(route, source_route):
    if source_route_get(source_route, "source_policy") == "law_first":
        return route

    return None


def search_all(query: str, k: int = 8, route=None, source_route=None, phrases=None):
    route = route if route is not None else route_query(query)
    source_route = source_route if source_route is not None else route_sources(query, total_k=k)
    phrases = phrases if phrases is not None else extract_legal_phrases(query, route=route)
    phapdien_route = route_for_phapdien(route, source_route)

    phapdien_vector = search_phapdien(query, k=50, route=phapdien_route)
    phapdien_fts = search_phapdien_fts(phrases, k=30)
    phapdien_res = rrf_merge(
        [phapdien_vector, phapdien_fts],
        weights=[1.0, 0.2],
    )
    
    anle = search_anle(query, k=30, route=route)
        
    evidence = phapdien_res + anle

    evidence = deduplicate_evidence(evidence)
    evidence = rerank_evidence(evidence, query)

    quotas = {
        "phapdien": source_route_get(source_route, "phapdien_k", 0),
        "anle": source_route_get(source_route, "anle_k", 0),
    }
    evidence = diversify_evidence(evidence, quotas)
    evidence = [expand_anle_evidence(expand_phapdien_evidence(item)) for item in evidence]
    evidence = order_evidence_by_policy(evidence, source_route_get(source_route, "source_policy"))

    return {
        "intent": source_route_get(source_route, "source_policy"),
        "quotas": quotas,
        "route": route,
        "source_route": source_route if isinstance(source_route, dict) else source_route.__dict__,
        "query_phrases": phrases,
        "evidence": evidence,
    }
