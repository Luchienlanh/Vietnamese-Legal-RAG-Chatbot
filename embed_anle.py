import os
import sqlite3
from pathlib import Path

os.environ["USE_TF"] = "0"

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from tqdm import tqdm


DATA_DIR = Path("data/huggingface/anle-toaan-gov-vn")
PERSIST_DIR = "data/vectorstores/anle_chroma_nvidia"
COLLECTION_NAME = "anle"
PARENT_DB = Path("data/index/anle_parents.sqlite")

MIN_CHARS = 80
MAX_PARAGRAPH_CHARS = 4000
WINDOW_SENTENCES = 5
WINDOW_OVERLAP = 1
EMBED_BATCH_SIZE = 32


def clean_value(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except ValueError:
        pass
    return str(value)


def clean_text(text):
    return " ".join(clean_value(text).split())


def load_document_metadata():
    path = DATA_DIR / "documents-00000-of-00001.parquet"

    columns = [
        "doc_name",
        "detail_url",
        "pdf_url",
        "doc_code",
        "doc_type",
        "case_type",
        "doc_subtype",
        "year",
        "title",
        "subject",
        "issue_date",
        "issuing_authority",
        "court_level",
        "jurisdiction",
        "precedent_number",
        "principle_text",
    ]

    df = pd.read_parquet(path, columns=columns)
    meta = {}

    for _, row in df.iterrows():
        doc_name = clean_value(row.get("doc_name"))
        if not doc_name:
            continue

        meta[doc_name] = {
            "doc_name": doc_name,
            "detail_url": clean_value(row.get("detail_url")),
            "pdf_url": clean_value(row.get("pdf_url")),
            "doc_code": clean_value(row.get("doc_code")),
            "doc_type": clean_value(row.get("doc_type")),
            "case_type": clean_value(row.get("case_type")),
            "doc_subtype": clean_value(row.get("doc_subtype")),
            "year": clean_value(row.get("year")),
            "title": clean_value(row.get("title")),
            "subject": clean_value(row.get("subject")),
            "issue_date": clean_value(row.get("issue_date")),
            "issuing_authority": clean_value(row.get("issuing_authority")),
            "court_level": clean_value(row.get("court_level")),
            "jurisdiction": clean_value(row.get("jurisdiction")),
            "precedent_number": clean_value(row.get("precedent_number")),
            "principle_text": clean_value(row.get("principle_text")),
        }

    return meta


def build_header(meta, page):
    title = meta.get("title") or meta.get("subject") or meta.get("doc_code")

    lines = [
        f"Tiêu đề: {title}",
        f"Số hiệu: {meta.get('doc_code', '')}",
        f"Loại văn bản: {meta.get('doc_type', '')}",
        f"Loại vụ án: {meta.get('case_type', '')}",
        f"Cấp tòa: {meta.get('court_level', '')}",
        f"Cơ quan ban hành: {meta.get('issuing_authority', '')}",
        f"Năm: {meta.get('year', '')}",
        f"Ngày ban hành: {meta.get('issue_date', '')}",
    ]

    if meta.get("precedent_number"):
        lines.append(f"Án lệ số: {meta.get('precedent_number')}")

    if page:
        lines.append(f"Trang PDF: {page}")

    return "\n".join(lines).strip()


def init_parent_db():
    PARENT_DB.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(PARENT_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parents (
            doc_name TEXT PRIMARY KEY,
            title TEXT,
            subject TEXT,
            doc_code TEXT,
            doc_type TEXT,
            case_type TEXT,
            doc_subtype TEXT,
            year TEXT,
            issue_date TEXT,
            issuing_authority TEXT,
            court_level TEXT,
            jurisdiction TEXT,
            detail_url TEXT,
            pdf_url TEXT,
            precedent_number TEXT,
            principle_text TEXT
        )
        """
    )
    conn.commit()
    return conn


def save_parents(conn, doc_meta):
    for meta in doc_meta.values():
        conn.execute(
            """
            INSERT OR REPLACE INTO parents (
                doc_name,
                title,
                subject,
                doc_code,
                doc_type,
                case_type,
                doc_subtype,
                year,
                issue_date,
                issuing_authority,
                court_level,
                jurisdiction,
                detail_url,
                pdf_url,
                precedent_number,
                principle_text
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                meta.get("doc_name", ""),
                meta.get("title", ""),
                meta.get("subject", ""),
                meta.get("doc_code", ""),
                meta.get("doc_type", ""),
                meta.get("case_type", ""),
                meta.get("doc_subtype", ""),
                meta.get("year", ""),
                meta.get("issue_date", ""),
                meta.get("issuing_authority", ""),
                meta.get("court_level", ""),
                meta.get("jurisdiction", ""),
                meta.get("detail_url", ""),
                meta.get("pdf_url", ""),
                meta.get("precedent_number", ""),
                meta.get("principle_text", ""),
            ),
        )

    conn.commit()


def paragraph_to_chunks(sentences, meta, paragraph_id, section_id):
    sentences = [clean_text(s) for s in sentences if clean_text(s)]
    if not sentences:
        return []

    paragraph_text = " ".join(sentences)
    if len(paragraph_text) < MIN_CHARS:
        return []

    page = meta.get("_page", "")
    header = build_header(meta, page)

    base_metadata = {
        "corpus": "anle",
        "doc_name": meta.get("doc_name", ""),
        "paragraph_id": paragraph_id,
        "section_id": section_id,
        "title": meta.get("title", ""),
        "subject": meta.get("subject", ""),
        "doc_code": meta.get("doc_code", ""),
        "doc_type": meta.get("doc_type", ""),
        "case_type": meta.get("case_type", ""),
        "doc_subtype": meta.get("doc_subtype", ""),
        "court_level": meta.get("court_level", ""),
        "year": meta.get("year", ""),
        "page": page,
        "detail_url": meta.get("detail_url", ""),
        "pdf_url": meta.get("pdf_url", ""),
        "precedent_number": meta.get("precedent_number", ""),
    }

    if len(paragraph_text) <= MAX_PARAGRAPH_CHARS:
        return [
            Document(
                page_content=f"{header}\n\n{paragraph_text}",
                metadata={**base_metadata, "chunk_index": 0},
            )
        ]

    docs = []
    step = WINDOW_SENTENCES - WINDOW_OVERLAP

    for start in range(0, len(sentences), step):
        window = sentences[start : start + WINDOW_SENTENCES]
        if not window:
            continue

        text = " ".join(window)
        if len(text) < MIN_CHARS:
            continue

        docs.append(
            Document(
                page_content=f"{header}\n\nĐoạn câu {start + 1}-{start + len(window)}:\n{text}",
                metadata={**base_metadata, "chunk_index": start},
            )
        )

    return docs


def load_chunks(doc_meta):
    chunks = []
    sentence_paths = sorted(DATA_DIR.glob("sentences-*.parquet"))

    columns = [
        "doc_name",
        "paragraph_id",
        "section_id",
        "page",
        "global_index",
        "text",
    ]

    for path in sentence_paths:
        df = pd.read_parquet(path, columns=columns)
        print(f"Processing {path} with {len(df)} rows")

        df = df.sort_values(["doc_name", "paragraph_id", "global_index"])
        grouped = df.groupby(["doc_name", "paragraph_id"], dropna=False)

        for (doc_name, paragraph_id), group in grouped:
            doc_name = clean_value(doc_name)
            meta = dict(doc_meta.get(doc_name, {"doc_name": doc_name}))

            pages = group["page"].dropna().unique().tolist()
            meta["_page"] = clean_value(pages[0]) if pages else ""

            section_ids = group["section_id"].dropna().unique().tolist()
            section_id = clean_value(section_ids[0]) if section_ids else ""

            sentences = group["text"].tolist()

            chunks.extend(
                paragraph_to_chunks(
                    sentences=sentences,
                    meta=meta,
                    paragraph_id=clean_value(paragraph_id),
                    section_id=section_id,
                )
            )

    return chunks


def make_chunk_id(doc):
    doc_name = doc.metadata.get("doc_name", "")
    paragraph_id = doc.metadata.get("paragraph_id", "")
    chunk_index = doc.metadata.get("chunk_index", 0)
    return f"{doc_name}:paragraph:{paragraph_id}:chunk:{chunk_index}"


def assign_unique_chunk_ids(chunks):
    seen = {}
    fixed = []

    for doc in chunks:
        base_id = make_chunk_id(doc)
        count = seen.get(base_id, 0)
        seen[base_id] = count + 1

        if count == 0:
            chunk_uid = base_id
        else:
            chunk_uid = f"{base_id}:dup:{count}"

        doc.metadata["chunk_uid"] = chunk_uid
        fixed.append(doc)

    duplicate_count = sum(count - 1 for count in seen.values() if count > 1)
    print(f"Duplicate chunk ids fixed: {duplicate_count}")

    return fixed


def main():
    load_dotenv()

    doc_meta = load_document_metadata()

    conn = init_parent_db()
    save_parents(conn, doc_meta)
    conn.close()

    chunks = load_chunks(doc_meta)
    chunks = assign_unique_chunk_ids(chunks)

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

    for i in tqdm(range(0, len(chunks), EMBED_BATCH_SIZE), desc="Embedding anle_v1_fixed"):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        ids = [doc.metadata["chunk_uid"] for doc in batch]
        vectorstore.add_documents(batch, ids=ids)

    print("Done.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Persist dir: {PERSIST_DIR}")
    print(f"Parent DB: {PARENT_DB}")


if __name__ == "__main__":
    main()