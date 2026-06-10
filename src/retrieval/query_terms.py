from __future__ import annotations

import re

from src.agents.base import parse_json_object
from src.agents.llm import get_llm as make_llm
from src.agents.phrase_extractor_agent import (
    PhraseExtractorAgent,
    clean_phrase,
    normalize_phrases,
)


def get_llm():
    return make_llm(max_tokens=300, temperature=0)


def build_phrase_prompt(query, route=None):
    return PhraseExtractorAgent().build_prompt(query, route=route)


def extract_legal_phrases(query, route=None, retries=3):
    return PhraseExtractorAgent().run(query, route=route, retries=retries)


def quote_fts_phrase(phrase):
    phrase = clean_phrase(phrase)
    phrase = phrase.replace('"', " ")
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return f'"{phrase}"'


def build_fts_query(phrases):
    phrases = normalize_phrases(phrases)
    if not phrases:
        return ""

    return " OR ".join(quote_fts_phrase(phrase) for phrase in phrases)
