import os
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

os.environ["USE_TF"] = "0"

load_dotenv()

ONTOLOGY_DIR = Path("data/huggingface/phapdien-moj-gov-vn")
REFERENCE_DB = Path("data/index/phapdien_reference_index.sqlite")
ROUTER_COLLECTION = "ontology_router"
ROUTER_STORE_DIR = "data/vectorstores/ontology_router_chroma_nvidia"
PHAPDIEN_COLLECTION = "phapdien"
PHAPDIEN_STORE_DIR = "data/vectorstores/phapdien_chroma_nvidia"


def get_embeddings():
    return NVIDIAEmbeddings(
        model=os.getenv("NVIDIA_EMBED_MODEL"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="END",
        max_batch_size=32,
    )


def safe_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def clean_metadata(**kwargs):
    return {
        key: safe_text(value)
        for key, value in kwargs.items()
    }


def build_ontology_documents():
    docs = []

    topics = pd.read_csv(ONTOLOGY_DIR / "ontology_topics.csv")
    for _, row in topics.iterrows():
        topic_title = safe_text(row.get("topic_title_vi"))
        topic_en = safe_text(row.get("topic_title_en"))
        topic_note = safe_text(row.get("topic_note"))

        docs.append(
            Document(
                page_content=(
                    f"Chủ đề pháp luật: {topic_title}\n"
                    f"English: {topic_en}\n"
                    f"Phạm vi: {topic_note}"
                ),
                metadata=clean_metadata(
                    router_id=f"topic:{row.get('topic_id')}",
                    node_kind="topic",
                    topic_id=row.get("topic_id"),
                    topic_number=row.get("topic_number"),
                    topic_title=topic_title,
                    subject_id="",
                    subject_title="",
                    glossary_category="",
                    label=topic_title,
                ),
            )
        )

    subjects = pd.read_csv(ONTOLOGY_DIR / "ontology_subjects.csv")
    for _, row in subjects.iterrows():
        topic_title = safe_text(row.get("topic_title_vi"))
        subject_title = safe_text(row.get("subject_title_vi"))
        subject_en = safe_text(row.get("subject_title_en"))

        docs.append(
            Document(
                page_content=(
                    f"Chủ đề pháp luật: {topic_title}\n"
                    f"Đề mục pháp luật: {subject_title}\n"
                    f"English: {subject_en}"
                ),
                metadata=clean_metadata(
                    router_id=f"subject:{row.get('subject_id')}",
                    node_kind="subject",
                    topic_id=row.get("topic_id"),
                    topic_number=row.get("topic_number"),
                    topic_title=topic_title,
                    subject_id=row.get("subject_id"),
                    subject_title=subject_title,
                    glossary_category="",
                    label=subject_title,
                ),
            )
        )

    glossary = pd.read_csv(ONTOLOGY_DIR / "ontology_glossary.csv")
    for _, row in glossary.iterrows():
        term = safe_text(row.get("vi"))
        term_en = safe_text(row.get("en"))
        category = safe_text(row.get("category"))
        note = safe_text(row.get("note"))

        docs.append(
            Document(
                page_content=(
                    f"Thuật ngữ pháp lý: {term}\n"
                    f"Nhóm: {category}\n"
                    f"English: {term_en}\n"
                    f"Ghi chú: {note}"
                ),
                metadata=clean_metadata(
                    router_id=f"glossary:{category}:{term}",
                    node_kind="glossary",
                    topic_id="",
                    topic_number="",
                    topic_title="",
                    subject_id="",
                    subject_title="",
                    glossary_category=category,
                    label=term,
                ),
            )
        )

    return docs


def load_router_store():
    return Chroma(
        collection_name=ROUTER_COLLECTION,
        persist_directory=ROUTER_STORE_DIR,
        embedding_function=get_embeddings(),
    )


def load_phapdien_store():
    return Chroma(
        collection_name=PHAPDIEN_COLLECTION,
        persist_directory=PHAPDIEN_STORE_DIR,
        embedding_function=get_embeddings(),
    )


def enrich_article_metadata(metadata):
    metadata = dict(metadata)
    if not REFERENCE_DB.exists():
        return metadata

    parent_uid = metadata.get("parent_uid")
    article_anchor = metadata.get("article_anchor")
    if not parent_uid and not article_anchor:
        return metadata

    conn = sqlite3.connect(REFERENCE_DB)
    conn.row_factory = sqlite3.Row

    row = None
    if parent_uid:
        row = conn.execute(
            "SELECT * FROM articles WHERE parent_uid = ? LIMIT 1",
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

    for key in [
        "topic_id",
        "topic_title",
        "subject_id",
        "subject_title",
        "article_title",
        "source_url",
    ]:
        metadata[key] = row[key]

    return metadata


def distance_to_similarity(distance):
    try:
        distance = float(distance)
    except (TypeError, ValueError):
        return 0.0

    if distance < 0:
        distance = 0.0

    return 1 / (1 + distance)


def normalize_match(doc, distance):
    metadata = dict(doc.metadata)
    return {
        "node_kind": metadata.get("node_kind", ""),
        "label": metadata.get("label", ""),
        "topic_id": metadata.get("topic_id", ""),
        "topic_title": metadata.get("topic_title", ""),
        "subject_id": metadata.get("subject_id", ""),
        "subject_title": metadata.get("subject_title", ""),
        "glossary_category": metadata.get("glossary_category", ""),
        "distance": float(distance),
        "similarity": distance_to_similarity(distance),
    }


def normalize_article_match(doc, distance):
    metadata = enrich_article_metadata(doc.metadata)
    return {
        "node_kind": "article",
        "label": metadata.get("article_title", ""),
        "topic_id": metadata.get("topic_id", ""),
        "topic_title": metadata.get("topic_title", ""),
        "subject_id": metadata.get("subject_id", ""),
        "subject_title": metadata.get("subject_title", ""),
        "glossary_category": "",
        "distance": float(distance),
        "similarity": distance_to_similarity(distance),
    }


def first_match(matches, node_kind):
    for match in matches:
        if match["node_kind"] == node_kind:
            return match
    return None


def route_query(query, k=8):
    vectorstore = load_router_store()
    phapdien_store = load_phapdien_store()

    if vectorstore._collection.count() == 0:
        return {
            "route_type": "ontology",
            "source_policy": "balanced",
            "confidence": 0.0,
            "topic": None,
            "subject": None,
            "glossary_hits": [],
            "matches": [],
        }

    ontology_results = vectorstore.similarity_search_with_score(query, k=k)
    ontology_matches = [
        normalize_match(doc, distance)
        for doc, distance in ontology_results
    ]

    article_results = phapdien_store.similarity_search_with_score(query, k=3)
    article_matches = [
        normalize_article_match(doc, distance)
        for doc, distance in article_results
    ]

    matches = article_matches + ontology_matches

    subject_match = first_match(article_matches, "article") or first_match(ontology_matches, "subject")
    topic_match = first_match(ontology_matches, "topic")

    if subject_match:
        topic = {
            "topic_id": subject_match["topic_id"],
            "topic_title": subject_match["topic_title"],
        }
        subject = {
            "subject_id": subject_match["subject_id"],
            "subject_title": subject_match["subject_title"],
        }
    elif topic_match:
        topic = {
            "topic_id": topic_match["topic_id"],
            "topic_title": topic_match["topic_title"],
        }
        subject = None
    else:
        topic = None
        subject = None

    glossary_hits = [
        {
            "term": match["label"],
            "category": match["glossary_category"],
            "similarity": match["similarity"],
        }
        for match in matches
        if match["node_kind"] == "glossary"
    ]

    confidence = matches[0]["similarity"] if matches else 0.0

    return {
        "route_type": "ontology",
        "source_policy": "balanced",
        "confidence": confidence,
        "topic": topic,
        "subject": subject,
        "glossary_hits": glossary_hits,
        "matches": matches,
    }
