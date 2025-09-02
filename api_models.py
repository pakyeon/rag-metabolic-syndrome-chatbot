from pydantic import BaseModel
from typing import List, Optional


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = "rag-gpt"
    messages: List[Message]
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = True


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatResponseChoice(BaseModel):
    index: int = 0
    message: ChatResponseMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = "rag-gpt"
    choices: List[ChatResponseChoice]
    usage: Usage


class SimpleAskRequest(BaseModel):
    question: str
    stream: Optional[bool] = True
