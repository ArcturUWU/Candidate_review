from typing import Any, List, Optional

import httpx

from ..config import settings


class LMStudioClient:
    """Minimal client for LM Studio HTTP API compatible with OpenAI chat format."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or settings.lm_studio_url
        self.client = httpx.Client(timeout=60)

    def chat(
        self,
        messages: List[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": settings.lm_model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
        resp = self.client.post(self.base_url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def stream_chat(
        self,
        messages: List[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
    ):
        payload: dict[str, Any] = {"model": settings.lm_model, "messages": messages, "temperature": temperature, "stream": True}
        if tools:
            payload["tools"] = tools
        with self.client.stream("POST", self.base_url, json=payload, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode()
                if not isinstance(line, str):
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line.split("data:", 1)[1].strip()
                if raw == "[DONE]":
                    break
                try:
                    import json as _json

                    parsed = _json.loads(raw)
                except Exception:
                    continue
                delta = parsed.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

    def ping(self) -> dict[str, Any]:
        """Lightweight connectivity check to LM Studio."""
        payload = {"model": settings.lm_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
        resp = self.client.post(self.base_url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()


lm_client = LMStudioClient()
