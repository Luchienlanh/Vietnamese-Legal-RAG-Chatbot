from pathlib import Path

import pyarrow.parquet as pq


DATASETS = {
    "phapdien": Path("data/huggingface/phapdien-moj-gov-vn"),
    "anle": Path("data/huggingface/anle-toaan-gov-vn"),
}


def count_rows(pattern: str) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in pattern)


def main() -> None:
    phapdien_articles = sorted(DATASETS["phapdien"].glob("articles-*.parquet"))
    anle_documents = sorted(DATASETS["anle"].glob("documents-*.parquet"))
    anle_sentences = sorted(DATASETS["anle"].glob("sentences-*.parquet"))
    anle_embed = sorted(DATASETS["anle"].glob("embed-*.parquet"))
    anle_reduce = sorted(DATASETS["anle"].glob("reduce-*.parquet"))

    print(f"phapdien articles: {count_rows(phapdien_articles):,} rows")
    print(f"anle documents: {count_rows(anle_documents):,} rows")
    print(f"anle sentences: {count_rows(anle_sentences):,} rows")
    print(f"anle embed: {count_rows(anle_embed):,} rows")
    print(f"anle reduce: {count_rows(anle_reduce):,} rows")


if __name__ == "__main__":
    main()
