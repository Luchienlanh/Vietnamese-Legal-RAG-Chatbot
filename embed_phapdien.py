import os
import sqlite3
import time
from pathlib import Path

os.environ["USE_TF"] = "0"

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


DATA_DIR = Path("data/huggingface/phapdien-moj-gov-vn")
PERSIST_DIR = "data/vectorstores/phapdien_chroma_nvidia"
COLLECTION_NAME = "phapdien"
PARENT_DB = Path("data/index/phapdien_parents.sqlite")

MAX_UNSPLIT_CHARS = 4000
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 250
EMBED_BATCH_SIZE = 16


def clean_value(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except ValueError:
        pass
    return str(value)


def load_ontology():
    topics = pd.read_csv(DATA_DIR / "ontology_topics.csv")
    subjects = pd.read_csv(DATA_DIR / "ontology_subjects.csv")
    glossary = pd.read_csv(DATA_DIR / "ontology_glossary.csv")

    topics_by_id = {
        clean_value(row.get("topic_id")): row.to_dict()
        for _, row in topics.iterrows()
    }

    subjects_by_id = {
        clean_value(row.get("subject_id")): row.to_dict()
        for _, row in subjects.iterrows()
    }

    instrument_terms = glossary[
        glossary["category"].astype(str).str.lower() == "instrument"
    ].to_dict("records")

    return topics_by_id, subjects_by_id, instrument_terms


def detect_instruments(source_note_text, instrument_terms):
    text = clean_value(source_note_text).lower()
    found = []

    for term in instrument_terms:
        vi = clean_value(term.get("vi"))
        en = clean_value(term.get("en"))
        note = clean_value(term.get("note"))

        if vi and vi.lower() in text:
            found.append(f"{vi} ({en}) - {note}")

    return found[:5]


def build_enriched_header(row, topics_by_id, subjects_by_id, instrument_terms):
    topic = topics_by_id.get(clean_value(row.get("topic_id")), {})
    subject = subjects_by_id.get(clean_value(row.get("subject_id")), {})
    instruments = detect_instruments(row.get("source_note_text"), instrument_terms)

    return f"""
Data source: Bo Phap Dien Viet Nam
Document type: codified Vietnamese legal article

Legal topic VI: {clean_value(row.get("topic_title"))}
Legal topic EN: {clean_value(topic.get("topic_title_en"))}
Topic note: {clean_value(topic.get("topic_note"))}

Legal subject VI: {clean_value(row.get("subject_title"))}
Legal subject EN: {clean_value(subject.get("subject_title_en"))}

Chapter: {clean_value(row.get("chapter_title"))}
Article title: {clean_value(row.get("article_title"))}
Source note: {clean_value(row.get("source_note_text"))}
Legal instrument types: {"; ".join(instruments)}
""".strip()


def init_parent_db():
    PARENT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(PARENT_DB)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parents (
            parent_uid TEXT PRIMARY KEY,
            article_anchor TEXT,
            article_title TEXT,
            topic_title TEXT,
            subject_title TEXT,
            chapter_title TEXT,
            source_url TEXT,
            source_note_text TEXT,
            content_text TEXT
        )
        """
    )
    conn.commit()
    return conn


def save_parent(conn, row, parent_uid):
    conn.execute(
        """
        INSERT OR REPLACE INTO parents (
            parent_uid,
            article_anchor,
            article_title,
            topic_title,
            subject_title,
            chapter_title,
            source_url,
            source_note_text,
            content_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            parent_uid,
            clean_value(row.get("article_anchor")),
            clean_value(row.get("article_title")),
            clean_value(row.get("topic_title")),
            clean_value(row.get("subject_title")),
            clean_value(row.get("chapter_title")),
            clean_value(row.get("source_url")),
            clean_value(row.get("source_note_text")),
            clean_value(row.get("content_text")),
        ),
    )


def make_child_docs(row, splitter, parent_uid, topics_by_id, subjects_by_id, instrument_terms):
    content = clean_value(row.get("content_text"))
    if not content.strip():
        return []

    article_anchor = clean_value(row.get("article_anchor"))
    if not article_anchor:
        return []

    header = build_enriched_header(row, topics_by_id, subjects_by_id, instrument_terms)

    metadata = {
        "parent_uid": parent_uid,
        "article_anchor": article_anchor,
        "corpus": "phapdien",
        "article_title": clean_value(row.get("article_title")),
        "topic_title": clean_value(row.get("topic_title")),
        "subject_title": clean_value(row.get("subject_title")),
        "chapter_title": clean_value(row.get("chapter_title")),
        "source_url": clean_value(row.get("source_url")),
        "source_note_text": clean_value(row.get("source_note_text")),
    }

    if len(content) <= MAX_UNSPLIT_CHARS:
        return [
            Document(
                page_content=f"{header}\n\nLegal article content:\n{content}",
                metadata={**metadata, "chunk_index": 0},
            )
        ]

    parts = splitter.split_text(content)
    return [
        Document(
            page_content=f"{header}\n\nChunk {i + 1} of legal article:\n{part}",
            metadata={**metadata, "chunk_index": i},
        )
        for i, part in enumerate(parts)
    ]


def make_chunk_id(doc):
    return f"{doc.metadata['parent_uid']}:chunk:{doc.metadata['chunk_index']}"


def assign_unique_chunk_ids(chunks):
    seen = {}
    for doc in chunks:
        base_id = make_chunk_id(doc)
        count = seen.get(base_id, 0)
        seen[base_id] = count + 1
        doc.metadata["chunk_uid"] = base_id if count == 0 else f"{base_id}:dup:{count}"

    duplicate_count = sum(count - 1 for count in seen.values() if count > 1)
    print(f"Duplicate chunk ids fixed: {duplicate_count}")
    return chunks


def load_chunks():
    topics_by_id, subjects_by_id, instrument_terms = load_ontology()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )

    conn = init_parent_db()
    chunks = []
    anchor_counts = {}

    for path in sorted(DATA_DIR.glob("articles-*.parquet")):
        df = pd.read_parquet(path)
        print(f"Processing {path} with {len(df)} rows")

        for _, row in df.iterrows():
            article_anchor = clean_value(row.get("article_anchor"))
            seen_count = anchor_counts.get(article_anchor, 0)
            anchor_counts[article_anchor] = seen_count + 1

            parent_uid = article_anchor if seen_count == 0 else f"{article_anchor}:dup:{seen_count}"

            save_parent(conn, row, parent_uid)
            chunks.extend(
                make_child_docs(
                    row,
                    splitter,
                    parent_uid,
                    topics_by_id,
                    subjects_by_id,
                    instrument_terms,
                )
            )

        conn.commit()

    conn.close()
    return assign_unique_chunk_ids(chunks)


def add_documents_with_retry(vectorstore, batch, ids, retries=5):
    for attempt in range(retries):
        try:
            vectorstore.add_documents(batch, ids=ids)
            return
        except Exception as exc:
            message = str(exc)
            if "502" not in message and "Bad Gateway" not in message:
                raise

            wait = 5 * (attempt + 1)
            print(f"502 from NVIDIA API. Retry {attempt + 1}/{retries} after {wait}s")
            time.sleep(wait)

    raise RuntimeError("Failed after retries due to NVIDIA API 502")

def existing_ids(vectorstore, ids):
    found = vectorstore._collection.get(ids=ids, include=[])
    return set(found.get("ids", []))

def main():
    load_dotenv()

    chunks = load_chunks()

    limit = int(os.getenv("PHAPDIEN_V3_LIMIT", "0") or "0")
    if limit > 0:
        chunks = chunks[:limit]

    print(f"Total chunks: {len(chunks)}")

    embeddings = NVIDIAEmbeddings(
        model=os.getenv("NVIDIA_EMBED_MODEL"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="END",
        max_batch_size=EMBED_BATCH_SIZE,
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    for i in tqdm(range(0, len(chunks), EMBED_BATCH_SIZE), desc="Embedding phapdien"):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        ids = [doc.metadata["chunk_uid"] for doc in batch]

        done_ids = existing_ids(vectorstore, ids)

        pending_batch = []
        pending_ids = []

        for doc, doc_id in zip(batch, ids):
            if doc_id in done_ids:
                continue
            pending_batch.append(doc)
            pending_ids.append(doc_id)

        if not pending_batch:
            continue

        add_documents_with_retry(vectorstore, pending_batch, pending_ids)

    print("Done.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Persist dir: {PERSIST_DIR}")
    print(f"Parent DB: {PARENT_DB}")


if __name__ == "__main__":
    main()