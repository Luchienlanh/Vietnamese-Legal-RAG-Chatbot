from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.context import format_memory


class MemoryRewriteAgent(BaseAgent):
    prompt_id = "memory_rewrite"

    def run(self, message: str, memory: dict) -> dict:
        memory_text = format_memory(memory)

        if not memory_text:
            return {
                "needs_memory": False,
                "need_memory": False,
                "rewritten_query": message,
                "reason": "No memory available.",
            }

        try:
            data = self.invoke_json(retries=3, memory=memory_text, question=message)
        except Exception:
            return {
                "needs_memory": False,
                "need_memory": False,
                "rewritten_query": message,
                "reason": "Rewrite failed, fallback to original query.",
            }

        rewritten_query = str(data.get("rewritten_query") or message).strip() or message
        needs_memory = bool(data.get("needs_memory", data.get("need_memory", False)))

        return {
            "needs_memory": needs_memory,
            "need_memory": needs_memory,
            "rewritten_query": rewritten_query,
            "reason": str(data.get("reason") or ""),
        }
