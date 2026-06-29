from __future__ import annotations

from typing import Any, List, Optional

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None  # type: ignore


class TokenCounter:
    def __init__(self, tokenizer_path: str, local_files_only: bool = True):
        self._tok = None
        if AutoTokenizer is None:
            return
        if not tokenizer_path.strip():
            return
        try:
            self._tok = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=local_files_only,
                use_fast=True,
            )
        except Exception:
            self._tok = None

    @staticmethod
    def normalize_messages_for_text(messages: List[Any]) -> str:
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role = str(m.get("role", "user"))
                content = str(m.get("content", ""))
            else:
                role = str(getattr(m, "role", "user"))
                content = str(getattr(m, "content", ""))
            parts.append(f"{role}: {content.strip()}")
        return "\n".join(parts)

    def count(self, messages: List[Any]) -> Optional[int]:
        if self._tok is None:
            return None
        try:
            text = self.normalize_messages_for_text(messages)
            return len(self._tok.encode(text, add_special_tokens=False))
        except Exception:
            return None

    def count_messages(self, messages: List[Any]) -> int:
        value = self.count(messages)
        return int(value) if value is not None else 0

    def count_messages_tokens(self, messages: List[Any]) -> int:
        value = self.count(messages)
        return int(value) if value is not None else 0

    def num_tokens_from_messages(self, messages: List[Any]) -> int:
        value = self.count(messages)
        return int(value) if value is not None else 0

    def count_text(self, text: str) -> int:
        if self._tok is None:
            return 0
        try:
            return len(self._tok.encode(str(text), add_special_tokens=False))
        except Exception:
            return 0

    def count_tokens(self, text: str) -> int:
        return self.count_text(text)

    def num_tokens(self, text: str) -> int:
        return self.count_text(text)
