"""OpenAI-compatible LLM client with zero-exposure API key loading, retries, and transcript logging."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_BASE_URL = "https://api.openai-next.com/v1"
DEFAULT_MODEL = "deepseek-v4-1-flash-261017"
DEFAULT_KEY_FILE = Path(
    r"D:\WARE_HOUSE\desktop_file\云大学科\网安实训中级\网安实训选题"
    r"\纯的api key（无任何符号，就是sk-……，使用的时候不要泄露，可以做无接触式复制粘贴）.txt"
)

_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


def redact_secrets(text: str) -> str:
    """Mask any secret key pattern in strings before logging or raising errors."""
    if not isinstance(text, str):
        return str(text)
    return _SECRET_PATTERN.sub("sk-***REDACTED***", text)


def _load_api_key() -> str:
    """Load API key from environment variable LLM_API_KEY, falling back to the read-only key file."""
    env_key = os.environ.get("LLM_API_KEY", "").strip()
    if env_key:
        return env_key
    if DEFAULT_KEY_FILE.exists():
        key = DEFAULT_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    raise RuntimeError(
        "LLM API key not found. Set LLM_API_KEY environment variable or ensure the key file exists."
    )


@dataclass
class LLMCallResult:
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""
    raw_message: Dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """Thin, auditable OpenAI Chat Completions client."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 90,
        max_retries: int = 4,
        transcript_path: Optional[Path] = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.transcript_path = Path(transcript_path) if transcript_path else None
        self._api_key = _load_api_key()

        # Running accounting
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.json_retries = 0

    def set_transcript_path(self, path: Optional[Path]) -> None:
        self.transcript_path = Path(path) if path else None

    def _append_transcript(self, record: Dict[str, Any]) -> None:
        if self.transcript_path is None:
            return
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = redact_secrets(json.dumps(record, ensure_ascii=False, default=str))
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(serialized + "\n")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2048,
        tag: str = "chat",
    ) -> LLMCallResult:
        """Send one chat completion request with exponential-backoff retries."""
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                latency = time.perf_counter() - t0
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    msg = choice.get("message", {})
                    usage = data.get("usage", {})
                    pt = int(usage.get("prompt_tokens", 0) or 0)
                    ct = int(usage.get("completion_tokens", 0) or 0)
                    tt = int(usage.get("total_tokens", pt + ct) or (pt + ct))

                    result = LLMCallResult(
                        content=msg.get("content") or "",
                        tool_calls=msg.get("tool_calls") or [],
                        finish_reason=choice.get("finish_reason") or "",
                        prompt_tokens=pt,
                        completion_tokens=ct,
                        total_tokens=tt,
                        latency_s=latency,
                        model=data.get("model", self.model),
                        raw_message=msg,
                    )
                    self.total_calls += 1
                    self.total_prompt_tokens += pt
                    self.total_completion_tokens += ct
                    self.total_tokens += tt
                    self.total_latency_s += latency

                    self._append_transcript(
                        {
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "tag": tag,
                            "model": result.model,
                            "request_messages_count": len(messages),
                            "last_request_message": messages[-1] if messages else None,
                            "response_content": result.content,
                            "response_tool_calls": result.tool_calls,
                            "finish_reason": result.finish_reason,
                            "usage": {
                                "prompt_tokens": pt,
                                "completion_tokens": ct,
                                "total_tokens": tt,
                            },
                            "latency_s": round(latency, 3),
                        }
                    )
                    return result

                err_body = redact_secrets(resp.text[:500])
                last_err = f"HTTP {resp.status_code}: {err_body}"
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(2 ** (attempt + 1), 16))
                    continue
                break
            except requests.RequestException as exc:
                last_err = redact_secrets(f"{type(exc).__name__}: {exc}")
                time.sleep(min(2 ** (attempt + 1), 16))

        self._append_transcript(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "tag": tag,
                "error": redact_secrets(last_err or "unknown error"),
            }
        )
        raise RuntimeError(f"LLM API request failed after {self.max_retries} attempts: {redact_secrets(last_err or '')}")

    def get_usage_summary(self) -> Dict[str, Any]:
        return {
            "calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_s": round(self.total_latency_s, 3),
            "json_retries": self.json_retries,
        }

    def reset_usage(self) -> None:
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.json_retries = 0


def extract_json_object(text: str) -> Dict[str, Any]:
    """Robustly parse a JSON object from LLM text (handles markdown code fences)."""
    if not text:
        raise ValueError("Empty LLM response; expected JSON object.")
    cleaned = text.strip()
    # Try direct parse first
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Try ```json ... ``` fence
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        obj = json.loads(fence.group(1))
        if isinstance(obj, dict):
            return obj
    # Fallback: outermost { ... }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        obj = json.loads(cleaned[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("Could not extract a valid JSON object from response.")
