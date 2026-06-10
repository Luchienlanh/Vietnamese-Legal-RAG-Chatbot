from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache

os.environ["USE_TF"] = "0"

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from src.prompting.loader import load_prompt

load_dotenv()


@dataclass
class SourceRoute:
    source_policy: str
    phapdien_k: int
    anle_k: int
    confidence: float
    matches: list
    reason: str


@lru_cache(maxsize=1)
def load_source_profiles() -> tuple[dict, ...]:
    profiles = load_prompt("source_router").get("profiles") or []
    return tuple(dict(profile) for profile in profiles)


def get_embeddings():
    return NVIDIAEmbeddings(
        model=os.getenv("NVIDIA_EMBED_MODEL"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="END",
        max_batch_size=32,
    )


def cosine_similarity(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


@lru_cache(maxsize=1)
def get_profile_embeddings():
    embeddings = get_embeddings()
    texts = [profile["description"].strip() for profile in load_source_profiles()]
    return embeddings.embed_documents(texts)


def rank_source_profiles(query):
    embeddings = get_embeddings()
    query_embedding = embeddings.embed_query(query)
    profile_embeddings = get_profile_embeddings()

    matches = []
    for profile, profile_embedding in zip(load_source_profiles(), profile_embeddings):
        matches.append(
            {
                "source_policy": profile["source_policy"],
                "score": cosine_similarity(query_embedding, profile_embedding),
            }
        )

    return sorted(matches, key=lambda item: item["score"], reverse=True)


def quotas_for_policy(source_policy, total_k):
    if source_policy == "law_first":
        anle_k = min(2, max(total_k - 1, 0))
        return total_k - anle_k, anle_k

    if source_policy == "case_first":
        phapdien_k = min(2, max(total_k - 1, 0))
        return phapdien_k, total_k - phapdien_k

    phapdien_k = total_k // 2
    return phapdien_k, total_k - phapdien_k


class SourceRouterAgent:
    def run(self, query: str, total_k: int = 8) -> SourceRoute:
        matches = rank_source_profiles(query)
        top_match = matches[0] if matches else {"source_policy": "balanced", "score": 0.0}
        source_policy = top_match["source_policy"]
        phapdien_k, anle_k = quotas_for_policy(source_policy, total_k)

        return SourceRoute(
            source_policy=source_policy,
            phapdien_k=phapdien_k,
            anle_k=anle_k,
            confidence=top_match["score"],
            matches=matches,
            reason="Selected by semantic similarity to source profiles.",
        )
