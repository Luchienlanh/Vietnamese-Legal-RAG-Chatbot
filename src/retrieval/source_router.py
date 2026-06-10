from __future__ import annotations

from src.agents.source_router_agent import (
    SourceRoute,
    SourceRouterAgent,
    cosine_similarity,
    get_embeddings,
    get_profile_embeddings,
    load_source_profiles,
    quotas_for_policy,
    rank_source_profiles,
)

SOURCE_PROFILES = load_source_profiles()


def route_sources(query: str, total_k: int = 8) -> SourceRoute:
    return SourceRouterAgent().run(query, total_k=total_k)
