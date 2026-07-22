"""LLM backend — DeepSeek primary (§13).

Primary: deepseek-v4-pro (thinking) / deepseek-v4-flash (cheap). No fallback.
"""
from __future__ import annotations

import json
import logging
import os
import random
import socket
import time
import urllib.error
import urllib.request
from typing import Any

_DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
_DEEPSEEK_MODEL = "deepseek-v4-pro"
_FLASH_MODEL = "deepseek-v4-flash"
_TIMEOUT = 180        # flash / cheap calls
_PRO_TIMEOUT = 1800   # thinking mode — up to 30 min

# Retry transient failures so concurrent callers (Phase 7 fans calls across threads)
# self-throttle on rate limits instead of silently losing results. Jittered backoff
# avoids a thundering herd when many workers get 429 at once.
_MAX_RETRIES = 5
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}

_log = logging.getLogger("erf.llm")


def _post_json(url: str, payload: dict, headers: dict, timeout: int = _TIMEOUT,
               *, max_retries: int = _MAX_RETRIES) -> dict:
    data = json.dumps(payload).encode()
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code not in _RETRYABLE_HTTP or attempt == max_retries:
                raise
            ra = e.headers.get("Retry-After") if e.headers else None
            base = float(ra) if (ra and str(ra).isdigit()) else min(2 ** attempt, 30)
            delay = base + random.uniform(0, 1.0)
            _log.warning("DeepSeek HTTP %s — retry %d/%d after %.1fs",
                         e.code, attempt + 1, max_retries, delay)
            time.sleep(delay)
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            if attempt == max_retries:
                raise
            delay = min(2 ** attempt, 30) + random.uniform(0, 1.0)
            _log.warning("DeepSeek conn error (%s) — retry %d/%d after %.1fs",
                         e, attempt + 1, max_retries, delay)
            time.sleep(delay)
    raise RuntimeError("unreachable")  # loop always returns or raises


def _strip_markdown_fences(text: str) -> str:
    import re
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*)\n?```\s*$", text, re.DOTALL)
    return m.group(1).strip() if m else text


def complete(messages: list[dict], *, json_only: bool = False, model: str | None = None,
             temperature: float = 0.0) -> str:
    """Send messages to DeepSeek and return the response text.

    model: None → deepseek-v4-pro (default / thinking).
           Pass 'deepseek-v4-flash' for cheap, fast, non-thinking passes.
    temperature > 0 for generation variety.
    Raises on failure — no silent fallback.
    """
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    used_model = model or _DEEPSEEK_MODEL
    payload: dict[str, Any] = {
        "model": used_model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    timeout = _PRO_TIMEOUT if used_model == _DEEPSEEK_MODEL else _TIMEOUT
    _log.debug("calling %s (timeout=%ds)", used_model, timeout)
    body = _post_json(_DEEPSEEK_URL, payload, headers, timeout=timeout)
    content = body["choices"][0]["message"]["content"]
    if json_only:
        content = _strip_markdown_fences(content)
    return content
