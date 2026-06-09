import sqlite3
from pathlib import Path

SOURCE_DB = Path(r"data\index\phapdien_reference_index.sqlite")
FTS_DB = Path(r"data\index\phapdien_fts.sqlite")


def main():
    FTS_DB.parent.mkdir(parents=True, exist_ok=True)
    
    src = sqlite3.connect(SOURCE_DB)
    src.row_factory = sqlite3.Row
    
    dst = sqlite3.connect(FTS_DB)
    
    dst.execute("DROP TABLE IF EXISTS phapdien_fts")
    dst.execute("""
            CREATE VIRTUAL TABLE phapdien_fts USING fts5(
            parent_uid UNINDEXED,
            article_anchor UNINDEXED,
            topic_id UNINDEXED,
            subject_id UNINDEXED,
            topic_title,
            subject_title,
            article_title,
            chapter_title,
            source_note_text,
            source_url UNINDEXED,
            content_text,
            tokenize='unicode61'
        )
    """)
    
    rows = src.execute("""
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
            content_text
        FROM articles
        WHERE content_text IS NOT NULL
          AND TRIM(content_text) != ''
    """)
    
    count = 0
    for row in rows:
        dst.execute("""
            INSERT INTO phapdien_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(row))
        count += 1

        if count % 5000 == 0:
            dst.commit()
            print("Inserted", count)

    dst.commit()
    src.close()
    dst.close()

    print("Done:", count)
    print("FTS DB:", FTS_DB)


if __name__ == "__main__":
    main()
        