from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from src.agents.llm import get_llm
from src.prompting.loader import get_model_config, load_prompt, render_prompt


class AgentError(RuntimeError):
    """Raised when an agent cannot complete its own step."""


def parse_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


def invoke_with_retry(llm, prompt: str, retries: int = 3, base_wait: int = 5):
    for attempt in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            message = str(exc)
            retryable = any(code in message for code in ["429", "502", "503", "504"])
            if not retryable or attempt == retries - 1:
                raise

            wait = base_wait * (attempt + 1)
            print(f"NVIDIA LLM temporary error. Retry {attempt + 1}/{retries} after {wait}s")
            time.sleep(wait)


class BaseAgent:
    prompt_id: str = ""

    def __init__(self, llm_factory: Callable[..., Any] = get_llm):
        if not self.prompt_id:
            raise AgentError(f"{self.__class__.__name__} must define prompt_id.")
        self.llm_factory = llm_factory

    @property
    def prompt_config(self) -> dict[str, Any]:
        return load_prompt(self.prompt_id)

    @property
    def model_config(self) -> dict[str, Any]:
        return get_model_config(self.prompt_id)

    def render_prompt(self, **values: Any) -> str:
        return render_prompt(self.prompt_id, **values)

    def make_llm(self, max_tokens: int | None = None, temperature: float | None = None):
        config = self.model_config
        tokens = int(max_tokens if max_tokens is not None else config.get("max_tokens", 1200))
        temp = float(temperature if temperature is not None else config.get("temperature", 0))
        return self.llm_factory(max_tokens=tokens, temperature=temp)

    def invoke_text(self, retries: int = 3, **values: Any) -> str:
        prompt = self.render_prompt(**values)
        response = invoke_with_retry(self.make_llm(), prompt, retries=retries)
        return str(response.content)

    def invoke_json(self, retries: int = 3, **values: Any) -> dict[str, Any]:
        return parse_json_object(self.invoke_text(retries=retries, **values))
