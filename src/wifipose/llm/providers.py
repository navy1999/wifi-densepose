"""Concrete LLM providers behind a single interface.

Free-tier first:
    - groq     Groq (Llama 3.x) — free, very low latency — recommended cloud choice
    - gemini   Google Gemini (Flash) — generous free tier, drop-in alternative
    - offline  no network, deterministic — default; powers CI & the live demo
Drop-in paid options (same interface):
    - anthropic / openai
"""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from wifipose.config import get_settings
from wifipose.llm.base import LLMProvider, LLMResult
from wifipose.logging_config import get_logger

log = get_logger("llm")

_DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
}


class OfflineProvider(LLMProvider):
    """No-network provider. NL->SQL and summaries use deterministic local logic
    (see nl_to_sql/summarizer), so `complete` is only a safe fallback."""

    name = "offline"

    @property
    def is_offline(self) -> bool:
        return True

    async def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=512) -> LLMResult:
        return LLMResult(
            text="[offline] LLM disabled; using deterministic rule-based logic.",
            provider=self.name,
            model="rule-based",
        )


class _HTTPProvider(LLMProvider):
    """Shared retry/timeout plumbing for HTTP-based providers."""

    def __init__(self, api_key: str, model: str, timeout: int):
        if not api_key:
            raise ValueError(f"{self.name} provider requires WIFIPOSE_LLM_API_KEY")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8), reraise=True)
    async def _post(self, url: str, *, headers: dict, json: dict) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=headers, json=json)
            resp.raise_for_status()
            return resp.json()


class GeminiProvider(_HTTPProvider):
    name = "gemini"

    async def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=512) -> LLMResult:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        data = await self._post(url, headers={"Content-Type": "application/json"}, json=body)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return LLMResult(text=text, provider=self.name, model=self.model)


class GroqProvider(_HTTPProvider):
    name = "groq"

    async def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=512) -> LLMResult:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = await self._post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        return LLMResult(
            text=data["choices"][0]["message"]["content"], provider=self.name, model=self.model
        )


class OpenAIProvider(GroqProvider):
    """OpenAI uses the same chat-completions schema as Groq."""

    name = "openai"

    async def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=512) -> LLMResult:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = await self._post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        return LLMResult(
            text=data["choices"][0]["message"]["content"], provider=self.name, model=self.model
        )


class AnthropicProvider(_HTTPProvider):
    name = "anthropic"

    async def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=512) -> LLMResult:
        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        data = await self._post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json=body,
        )
        return LLMResult(text=data["content"][0]["text"], provider=self.name, model=self.model)


_REGISTRY = {
    "offline": OfflineProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider() -> LLMProvider:
    """Construct the configured provider, falling back to offline on misconfig."""
    s = get_settings()
    key = s.llm_provider.lower().strip()
    if key not in _REGISTRY:
        log.warning("unknown_llm_provider", provider=key, fallback="offline")
        key = "offline"
    if key == "offline":
        return OfflineProvider()
    model = s.llm_model or _DEFAULT_MODELS.get(key, "")
    try:
        return _REGISTRY[key](api_key=s.llm_api_key, model=model, timeout=s.llm_timeout)
    except ValueError as e:
        log.warning("llm_provider_init_failed", provider=key, error=str(e), fallback="offline")
        return OfflineProvider()
