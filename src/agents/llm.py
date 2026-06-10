from __future__ import annotations

import os

os.environ["USE_TF"] = "0"

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()


def get_llm(max_tokens: int = 1200, temperature: float = 0):
    return ChatNVIDIA(
        model=os.getenv("NVIDIA_LLM_MODEL"),
        api_key=os.getenv("NVIDIA_API_LLM") or os.getenv("NVIDIA_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
    )
