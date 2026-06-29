from typing import Any, Dict, List, Tuple, Protocol


class IRAG(Protocol):
    kb_version: str

    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        ...

    def format_context(self, items: List[Dict[str, Any]]) -> str:
        ...
