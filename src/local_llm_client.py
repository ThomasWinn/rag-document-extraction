"""
Utility helpers for talking to a local LM Studio server that exposes
OpenAI-compatible endpoints.

All configuration is sourced from environment variables so that the same code
works with different local models or ports:

* ``LM_STUDIO_BASE_URL`` (default: ``http://127.0.0.1:1234``)
* ``LM_STUDIO_MODEL_ID`` (default: ``qwen2.5-32b-instruct-mlx``)
* ``LM_STUDIO_API_KEY`` (optional: sent as bearer token if provided)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional

import requests


def _resolve_base_url() -> str:
    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234")
    return base_url.rstrip("/")


def _resolve_model_id() -> str:
    return os.getenv("LM_STUDIO_MODEL_ID", "qwen2.5-32b-instruct-mlx")


def generate_chat_completion(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    stop: Optional[Iterable[str]] = None,
    stream: bool = False,
    timeout: int = 120,
) -> str:
    """
    Issue a chat completion request to the LM Studio server and return the
    assistant's response text.

    Args:
        prompt: User message sent to the model.
        system_prompt: Optional system prompt that precedes the user message.
        temperature: Sampling temperature.
        max_tokens: Maximum number of tokens to generate.
        stop: Optional collection of stop sequences.
        stream: When True, the LM Studio server streams partial responses. This
            helper currently aggregates the final text, so streaming is disabled
            by default.
        timeout: Timeout in seconds for the HTTP request.
    """

    messages: list[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": _resolve_model_id(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if stop:
        payload["stop"] = list(stop)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('LM_STUDIO_API_KEY', 'lm-studio')}",
    }

    url = f"{_resolve_base_url()}/v1/chat/completions"
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()

    try:
        data = response.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError("LM Studio response was not valid JSON.") from exc

    choices = data.get("choices")
    if not choices:
        raise RuntimeError("LM Studio response did not include any choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise RuntimeError("LM Studio response did not include message content.")

    return content


__all__ = ["generate_chat_completion"]
