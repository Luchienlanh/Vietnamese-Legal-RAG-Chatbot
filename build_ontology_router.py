import time

from langchain_chroma import Chroma
from tqdm import tqdm

from src.retrieval.ontology_router import (
    ROUTER_COLLECTION,
    ROUTER_STORE_DIR,
    build_ontology_documents,
    get_embeddings,
)


BATCH_SIZE = 32
RETRIES = 5


def add_with_retry(vectorstore, docs, ids):
    for attempt in range(RETRIES):
        try:
            vectorstore.add_documents(docs, ids=ids)
            return
        except Exception as exc:
            message = str(exc)
            retryable = any(code in message for code in ["429", "502", "503", "504"])
            if not retryable or attempt == RETRIES - 1:
                raise

            wait = 5 * (attempt + 1)
            print(f"NVIDIA API temporary error. Retry {attempt + 1}/{RETRIES} after {wait}s")
            time.sleep(wait)


def main():
    docs = build_ontology_documents()
    vectorstore = Chroma(
        collection_name=ROUTER_COLLECTION,
        persist_directory=ROUTER_STORE_DIR,
        embedding_function=get_embeddings(),
    )
    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    vectorstore = Chroma(
        collection_name=ROUTER_COLLECTION,
        persist_directory=ROUTER_STORE_DIR,
        embedding_function=get_embeddings(),
    )

    fixed_docs = []
    fixed_ids = []
    seen = set()

    for doc in docs:
        router_id = doc.metadata["router_id"]
        if router_id in seen:
            continue

        seen.add(router_id)
        fixed_docs.append(doc)
        fixed_ids.append(router_id)

    for start in tqdm(range(0, len(fixed_docs), BATCH_SIZE), desc="Embedding ontology router"):
        end = start + BATCH_SIZE
        add_with_retry(
            vectorstore,
            fixed_docs[start:end],
            fixed_ids[start:end],
        )

    print(f"Ontology router documents: {len(fixed_docs)}")
    print(f"Chroma collection count: {vectorstore._collection.count()}")
    print(f"Persist directory: {ROUTER_STORE_DIR}")


if __name__ == "__main__":
    main()
