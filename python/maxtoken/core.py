from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
    # Stop strings (OpenAI `stop` / Anthropic `stop_sequences`). Generation finishes when one
    # appears in the decoded output; the matched substring (and anything after) is trimmed.
    stop_strs: list[str] = field(default_factory=list)
    # Structured output (response_format), decoded under a grammar on the MLX
    # backend: {"type": "json_object" | "json_schema", "schema": {...},
    # "strict": bool, "start_after": "</think>" for a thinking model}. None = free text.
    grammar: dict | None = None

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
