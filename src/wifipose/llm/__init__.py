"""LLM analytics layer.

Provider-agnostic interface (`LLMProvider`) with free-tier implementations
(Groq, Google Gemini) plus a zero-dependency `OfflineProvider` so the platform —
and CI, and the live demo — works with no API key at all. Anthropic Claude and
OpenAI drop in behind the same interface via one env var.

Two product features sit on top:
    - natural-language -> SQL analytics over the pose_events time-series
    - session activity summaries
"""

from wifipose.llm.base import LLMProvider, LLMResult
from wifipose.llm.providers import get_provider

__all__ = ["LLMProvider", "LLMResult", "get_provider"]
