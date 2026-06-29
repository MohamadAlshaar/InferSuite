from typing import List, Literal, Optional
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    model: Optional[str] = None
    bypass_rag: Optional[bool] = False
    bypass_cache: Optional[bool] = False
    session_id: Optional[str] = None

