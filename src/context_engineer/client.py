"""OpenAI-compatible HTTP client for LM Studio."""

import logging
from typing import Any

import httpx

from context_engineer.types import Message

logger = logging.getLogger(__name__)


class LMStudioClient:
    """Client for local OpenAI-compatible inference servers like LM Studio."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def is_available(self) -> bool:
        """Checks whether the LM Studio server is reachable."""
        try:
            url = f"{self.base_url}/models"
            resp = httpx.get(url, headers=self._headers, timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Retrieves list of loaded models from the local server."""
        try:
            url = f"{self.base_url}/models"
            resp = httpx.get(url, headers=self._headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.warning("Failed to list models from %s: %s", self.base_url, e)
            return []

    def chat_completion(
        self,
        messages: list[Message],
        model: str = "qwen/qwen3.5-9b",
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] | None = None,
    ) -> str:
        """Dispatches a chat completion request to the LM Studio endpoint."""
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        resp = httpx.post(url, json=payload, headers=self._headers, timeout=self.timeout)
        resp.raise_for_status()
        result = resp.json()

        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices returned from LM Studio response: {result}")

        content = choices[0].get("message", {}).get("content", "")
        return str(content)
