import json
import re
import sqlite3
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/huggingface/phapdien-moj-gov-vn")
DB_PATH = Path("data/index/phapdien_reference_index.sqlite")


def clean_value(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except ValueError:
        pass
    return str(value)


def parse_source_links(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if hasattr(value, "tolist"):
        return value.tolist()

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []

    return []


def extract_item_id_from_href(href):
    href = clean_value(href)
    match = re.search(r"ItemID=(\d+)", href)
    if not match:
        return ""
    return match.group(1)


def extract_article_number_from_source_note(text):
    text = clean_value(text)
    match = re.search(r"Điều\s+(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1)


def extract_instrument_key(source_note_text):
    text = clean_value(source_note_text)

    patterns = [
        r"(Luật\s+số\s+[0-9A-Za-z/.-]+)",
        r"(Bộ luật\s+[A-Za-zÀ-ỹ\s]+(?:\d{4})?)",
        r"(Nghị định\s+số\s+[0-9A-Za-z/.-]+)",
        r"(Thông tư\s+số\s+[0-9A-Za-z/.-]+)",
        r"(Thông tư liên tịch\s+số\s+[0-9A-Za-z/.-]+)",
        r"(Nghị quyết\s+số\s+[0-9A-Za-z/.-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return " ".join(match.group(1).split())

    return ""


def extract_primary_source_item_id(source_links):
    for link in source_links:
        if not isinstance(link, dict):
            continue
        item_id = extract_item_id_from_href(link.get("href"))
        if item_id:
            return item_id
    return ""


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS articles")

    conn.execute(
        """
        CREATE TABLE articles (
            parent_uid TEXT PRIMARY KEY,
            article_anchor TEXT,
            topic_id TEXT,
            subject_id TEXT,
            topic_title TEXT,
            subject_title TEXT,
            article_title TEXT,
            chapter_title TEXT,
            source_note_text TEXT,
            source_url TEXT,
            content_text TEXT,
            source_item_id TEXT,
            original_article_number TEXT,
            source_instrument_key TEXT
        )
        """
    )

    conn.execute("CREATE INDEX idx_source_article ON articles(source_item_id, original_article_number)")
    conn.execute("CREATE INDEX idx_subject_article ON articles(subject_id, original_article_number)")
    conn.execute("CREATE INDEX idx_topic_article ON articles(topic_id, original_article_number)")
    conn.execute("CREATE INDEX idx_instrument_article ON articles(source_instrument_key, original_article_number)")

    conn.commit()
    return conn


def insert_article(conn, row, parent_uid):
    source_links = parse_source_links(row.get("source_links"))
    source_item_id = extract_primary_source_item_id(source_links)
    original_article_number = extract_article_number_from_source_note(row.get("source_note_text"))
    source_instrument_key = extract_instrument_key(row.get("source_note_text"))

    conn.execute(
        """
        INSERT OR REPLACE INTO articles (
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
            source_item_id,
            original_article_number,
            source_instrument_key
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            parent_uid,
            clean_value(row.get("article_anchor")),
            clean_value(row.get("topic_id")),
            clean_value(row.get("subject_id")),
            clean_value(row.get("topic_title")),
            clean_value(row.get("subject_title")),
            clean_value(row.get("article_title")),
            clean_value(row.get("chapter_title")),
            clean_value(row.get("source_note_text")),
            clean_value(row.get("source_url")),
            clean_value(row.get("content_text")),
            source_item_id,
            original_article_number,
            source_instrument_key,
        ),
    )


def main():
    conn = init_db()

    anchor_counts = {}
    total = 0

    columns = [
        "article_anchor",
        "topic_id",
        "subject_id",
        "topic_title",
        "subject_title",
        "article_title",
        "chapter_title",
        "source_note_text",
        "source_links",
        "source_url",
        "content_text",
    ]

    for path in sorted(DATA_DIR.glob("articles-*.parquet")):
        df = pd.read_parquet(path, columns=columns)
        print(f"Processing {path} with {len(df)} rows")

        for _, row in df.iterrows():
            article_anchor = clean_value(row.get("article_anchor"))
            seen_count = anchor_counts.get(article_anchor, 0)
            anchor_counts[article_anchor] = seen_count + 1

            parent_uid = article_anchor if seen_count == 0 else f"{article_anchor}:dup:{seen_count}"

            insert_article(conn, row, parent_uid)
            total += 1

        conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    with_item_id = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE source_item_id != ''"
    ).fetchone()[0]
    with_article_number = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE original_article_number != ''"
    ).fetchone()[0]

    conn.close()

    print("Done.")
    print(f"Rows read: {total}")
    print(f"Rows indexed: {count}")
    print(f"With source_item_id: {with_item_id}")
    print(f"With original_article_number: {with_article_number}")
    print(f"DB: {DB_PATH}")


if __name__ == "__main__":
    main()