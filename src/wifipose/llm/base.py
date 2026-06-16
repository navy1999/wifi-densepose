"""LLM provider interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str


class LLMProvider(abc.ABC):
    """Minimal text-completion interface every backend implements."""

    name: str = "base"

    @abc.abstractmethod
    async def complete(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.0, max_tokens: int = 512
    ) -> LLMResult:
        """Return a single completion for `prompt`."""

    @property
    def is_offline(self) -> bool:
        return False
