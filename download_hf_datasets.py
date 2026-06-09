from pathlib import Path

from huggingface_hub import snapshot_download


BASE_DIR = Path("data") / "huggingface"


DATASETS = [
    {
        "repo_id": "tmquan/phapdien-moj-gov-vn",
        "local_dir": BASE_DIR / "phapdien-moj-gov-vn",
        "allow_patterns": [
            "README.md",
            "analytics.json",
            "ontology.json",
            "ontology_glossary.*",
            "ontology_subjects.*",
            "ontology_topics.*",
            "subjects.parquet",
            "tree_nodes.parquet",
            "articles-*.parquet",
        ],
    },
    {
        "repo_id": "tmquan/anle-toaan-gov-vn",
        "local_dir": BASE_DIR / "anle-toaan-gov-vn",
        "allow_patterns": [
            "README.md",
            "_stats.json",
            "manifest.json",
            "documents-*.parquet",
            "sentences-*.parquet",
            "embed-*.parquet",
            "reduce-*.parquet",
        ],
    },
]


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        repo_id = dataset["repo_id"]
        local_dir = dataset["local_dir"]
        print(f"Downloading {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=dataset["allow_patterns"],
            max_workers=4,
        )

    print("Done.")


if __name__ == "__main__":
    main()
