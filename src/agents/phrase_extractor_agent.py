from __future__ import annotations

import re

from src.agents.base import BaseAgent


def clean_phrase(phrase):
    phrase = str(phrase).lower()
    phrase = re.sub(r"[^\wÀ-ỹ\s]", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase


def normalize_phrases(phrases, max_phrases=8):
    if isinstance(phrases, str):
        phrases = [phrases]

    output = []
    seen = set()

    for phrase in phrases or []:
        phrase = clean_phrase(phrase)
        if not phrase:
            continue

        if len(phrase.split()) > 10:
            continue

        if phrase in seen:
            continue

        seen.add(phrase)
        output.append(phrase)

        if len(output) >= max_phrases:
            break

    return output


class PhraseExtractorAgent(BaseAgent):
    prompt_id = "phrase_extractor"

    def build_prompt(self, query: str, route=None) -> str:
        topic = ((route or {}).get("topic") or {}).get("topic_title", "")
        subject = ((route or {}).get("subject") or {}).get("subject_title", "")
        return self.render_prompt(question=query, topic=topic, subject=subject)

    def run(self, query: str, route=None, retries: int = 3) -> list[str]:
        topic = ((route or {}).get("topic") or {}).get("topic_title", "")
        subject = ((route or {}).get("subject") or {}).get("subject_title", "")

        try:
            data = self.invoke_json(
                retries=retries,
                question=query,
                topic=topic,
                subject=subject,
            )
        except Exception:
            return []

        phrases = data.get("phrases", [])
        if not isinstance(phrases, list):
            return []

        return normalize_phrases(phrases)
