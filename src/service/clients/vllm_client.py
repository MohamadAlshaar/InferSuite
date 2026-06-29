from __future__ import annotations

import copy
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Tuple

import httpx


def _normalize_path(path: str, default: str) -> str:
    raw = (path or default).strip()
    if not raw:
        raw = default
    if not raw.startswith("/"):
        raw = "/" + raw
    return raw


class VLLMClient:
    """
    Phase 1 backend adapter.

    - direct_vllm:
        sends OpenAI-style chat payloads directly to /v1/chat/completions
    - llmd + chat:
        sends the same chat payload shape to the configured llm-d chat path
    - llmd + completions:
        converts chat messages into a prompt and sends to the configured
        llm-d completions path, then normalizes the response back into a
        chat-completions-shaped dict for the rest of the service.
    """

    def __init__(
        self,
        *,
        backend: str,
        direct_vllm_base_url: str,
        llmd_base_url: str,
        llmd_api_mode: str = "chat",
        llmd_chat_path: str = "/v1/chat/completions",
        llmd_completions_path: str = "/v1/completions",
        served_model_name: str = "",
        timeout_s: float = 300.0,
    ):
        self.backend = (backend or "direct_vllm").strip().lower()
        if self.backend not in {"direct_vllm", "llmd"}:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.llmd_api_mode = (llmd_api_mode or "chat").strip().lower()
        if self.llmd_api_mode not in {"chat", "completions"}:
            raise ValueError(f"Unsupported llm-d api mode: {self.llmd_api_mode}")

        self.direct_vllm_base_url = (direct_vllm_base_url or "").rstrip("/")
        self.llmd_base_url = (llmd_base_url or "").rstrip("/")
        self.llmd_chat_path = _normalize_path(llmd_chat_path, "/v1/chat/completions")
        self.llmd_completions_path = _normalize_path(llmd_completions_path, "/v1/completions")
        self.served_model_name = (served_model_name or "").strip()

        if self.backend == "direct_vllm":
            base_url = self.direct_vllm_base_url
        else:
            base_url = self.llmd_base_url

        if not base_url:
            raise ValueError(
                f"Missing base URL for backend={self.backend}. "
                f"Set VLLM_BASE_URL or LLMD_BASE_URL accordingly."
            )

        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def create_chat_completion(
        self, payload: Dict[str, Any], request_id: str
    ) -> Dict[str, Any]:
        return await self.chat_completions(payload, request_id=request_id)

    async def chat_completions(
        self, payload: Dict[str, Any], request_id: str
    ) -> Dict[str, Any]:
        request_path, request_payload = self._prepare_request(payload)

        # Retry transient malformed/empty backend responses. Under heavy load
        # (e.g. all pods being perf-profiled) the backend occasionally returns a
        # non-JSON or truncated body; without a retry a single transient hiccup
        # becomes a hard "Invalid JSON" error with 0 output tokens (observed on
        # SC cache-miss generation). Retry the POST on parse failure or 5xx.
        max_attempts = 3
        t0 = time.perf_counter()
        r = None
        data: Dict[str, Any] = {"error": "Invalid JSON from model backend"}
        for attempt in range(max_attempts):
            r = await self._client.post(
                request_path,
                json=request_payload,
                headers={"X-Request-Id": request_id},
            )
            if r.status_code >= 500 and attempt < max_attempts - 1:
                continue
            try:
                data = r.json()
                break
            except Exception:
                if attempt < max_attempts - 1:
                    continue
                data = {"error": "Invalid JSON from model backend", "text": r.text,
                        "_parse_retries": attempt + 1}
        http_ms = (time.perf_counter() - t0) * 1000.0
        json_ms = 0.0

        if not isinstance(data, dict):
            return {
                "error": "Model backend returned non-dict JSON",
                "_http_status": r.status_code,
                "_http_ms": http_ms,
                "_json_ms": json_ms,
                "_backend": self.backend,
                "_backend_path": request_path,
            }

        if self.backend == "llmd" and self.llmd_api_mode == "completions":
            data = self._normalize_completion_response_to_chat(
                data,
                requested_model=str(payload.get("model") or self.served_model_name or ""),
            )

        data["_http_status"] = r.status_code
        data["_http_ms"] = http_ms
        data["_json_ms"] = json_ms
        data["_backend"] = self.backend
        data["_backend_path"] = request_path
        data["_backend_api_mode"] = (
            self.llmd_api_mode if self.backend == "llmd" else "chat"
        )
        return data

    async def stream_chat_completions(
        self, payload: Dict[str, Any], request_id: str
    ) -> AsyncIterator[bytes]:
        request_path, request_payload = self._prepare_request(payload)

        async with self._client.stream(
            "POST",
            request_path,
            json=request_payload,
            headers={
                "Accept": "text/event-stream",
                "X-Request-Id": request_id,
            },
        ) as r:
            async for line in r.aiter_lines():
                if line:
                    yield (line + "\n").encode("utf-8")

    def _prepare_request(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if self.backend == "direct_vllm":
            path, req = "/v1/chat/completions", copy.deepcopy(payload)
        elif self.llmd_api_mode == "chat":
            path, req = self.llmd_chat_path, copy.deepcopy(payload)
        else:
            path, req = self.llmd_completions_path, self._chat_payload_to_completion_payload(payload)
        self._force_exact_tokens(req)
        return path, req

    @staticmethod
    def _force_exact_tokens(req: Dict[str, Any]) -> None:
        """Benchmark mode: force vLLM to emit EXACTLY max_tokens (no early EOS),
        so output length is a controlled variable. Enabled via env so normal
        serving is unaffected. vLLM honours ignore_eos + min_tokens."""
        if os.getenv("VLLM_FORCE_EXACT_TOKENS") != "1":
            return
        mt = req.get("max_tokens")
        if mt:
            req["ignore_eos"] = True
            req["min_tokens"] = int(mt)

    def _chat_payload_to_completion_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "model": payload.get("model") or self.served_model_name,
            "prompt": self._messages_to_prompt(payload.get("messages")),
        }

        for key in (
            "max_tokens",
            "min_tokens",
            "ignore_eos",
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "stream",
            "seed",
            "n",
            "best_of",
        ):
            if key in payload and payload.get(key) is not None:
                out[key] = payload.get(key)

        return out

    def _messages_to_prompt(self, messages: Any) -> str:
        if not isinstance(messages, list):
            return ""

        role_map = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
        }

        lines: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = role_map.get(str(msg.get("role") or "user").strip().lower(), "User")
            content = msg.get("content")

            if isinstance(content, list):
                text_parts: List[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text") or ""))
                content = "\n".join(text_parts)

            text = str(content or "").strip()
            if not text:
                continue

            lines.append(f"{role}: {text}")

        if not lines:
            return ""

        lines.append("Assistant:")
        return "\n\n".join(lines)

    def _normalize_completion_response_to_chat(
        self,
        data: Dict[str, Any],
        *,
        requested_model: str,
    ) -> Dict[str, Any]:
        out = copy.deepcopy(data)

        choices = out.get("choices")
        if not isinstance(choices, list):
            out.setdefault("id", f"chatcmpl-{uuid.uuid4()}")
            out.setdefault("object", "chat.completion")
            out.setdefault("created", int(time.time()))
            out["model"] = out.get("model") or requested_model or self.served_model_name
            return out

        normalized_choices: List[Dict[str, Any]] = []
        for idx, choice in enumerate(choices):
            if not isinstance(choice, dict):
                normalized_choices.append(
                    {
                        "index": idx,
                        "message": {"role": "assistant", "content": str(choice)},
                        "finish_reason": None,
                    }
                )
                continue

            content = self._extract_choice_content(choice)
            normalized_choice: Dict[str, Any] = {
                "index": choice.get("index", idx),
                "message": {"role": "assistant", "content": content},
                "finish_reason": choice.get("finish_reason"),
            }
            if "logprobs" in choice:
                normalized_choice["logprobs"] = choice["logprobs"]
            normalized_choices.append(normalized_choice)

        out["choices"] = normalized_choices
        out["id"] = out.get("id") or f"chatcmpl-{uuid.uuid4()}"
        out["object"] = "chat.completion"
        out["created"] = int(out.get("created") or time.time())
        out["model"] = out.get("model") or requested_model or self.served_model_name
        return out

    def _extract_choice_content(self, choice: Dict[str, Any]) -> str:
        message = choice.get("message")
        if isinstance(message, dict):
            return str(message.get("content") or "")

        if "text" in choice:
            return str(choice.get("text") or "")

        if "content" in choice:
            return str(choice.get("content") or "")

        return ""
