from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - depends on runtime environment
    yaml = None

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class PromptError(ValueError):
    """Raised when a prompt file is missing, invalid, or rendered incorrectly."""


@lru_cache(maxsize=None)
def load_prompt(prompt_id: str) -> dict[str, Any]:
    path = PROMPTS_DIR / f"{prompt_id}.yaml"
    if not path.exists():
        raise PromptError(f"Prompt file not found for '{prompt_id}': {path}")

    text = path.read_text(encoding="utf-8")

    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PromptError(
                "PyYAML is required to load non-JSON YAML prompt files. "
                "Install it with: pip install pyyaml"
            ) from exc

    if not isinstance(data, dict):
        raise PromptError(f"Prompt '{prompt_id}' must be a YAML object.")

    declared_id = data.get("id")
    if declared_id and declared_id != prompt_id:
        raise PromptError(
            f"Prompt id mismatch: expected '{prompt_id}', got '{declared_id}'."
        )

    return data


def get_model_config(prompt_id: str) -> dict[str, Any]:
    config = load_prompt(prompt_id)
    model = config.get("model") or {}
    if not isinstance(model, dict):
        raise PromptError(f"Prompt '{prompt_id}' field 'model' must be an object.")
    return model


def render_prompt(prompt_id: str, **values: Any) -> str:
    config = load_prompt(prompt_id)
    required = config.get("input_variables") or []

    missing = [name for name in required if name not in values]
    if missing:
        joined = ", ".join(missing)
        raise PromptError(f"Prompt '{prompt_id}' missing input variables: {joined}")

    system = str(config.get("system") or "").strip()
    template = str(config.get("template") or "").strip()

    try:
        rendered_template = template.format(**values).strip() if template else ""
    except KeyError as exc:
        raise PromptError(f"Prompt '{prompt_id}' missing template variable: {exc}") from exc

    return "\n\n".join(part for part in [system, rendered_template] if part).strip()
