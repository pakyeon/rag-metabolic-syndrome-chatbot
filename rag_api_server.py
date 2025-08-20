from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import asyncio

import time
import uuid
import os
import json

# ------------------------------------------------------------------
# FastAPI 앱 설정
# ------------------------------------------------------------------
app = FastAPI(
    title="RAG API Server",
    description="FastAPI로 제공하는 RAG(검색증강생성) 질문-답변 서비스",
    version="0.2.3",
)

# ------------------------------------------------------------------
# CORS (OpenWebUI 등 브라우저 클라이언트 허용)
# ------------------------------------------------------------------
default_origins = ["http://localhost:8080", "http://127.0.0.1:8080"]
env_origins = os.getenv("OPENWEBUI_ORIGINS", "")
allow_origins = (
    [o.strip() for o in env_origins.split(",") if o.strip()]
    if env_origins
    else default_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# RAG 엔진 지연 로딩
# ------------------------------------------------------------------
_rag_engine = None
_rag_app = None


def get_rag_engine():
    """RAG 엔진을 지연 로딩"""
    global _rag_engine, _rag_app
    if _rag_engine is None:
        print("[INFO] Loading RAG engine (first request)...")
        from rag_engine_v3 import answer_question_graph, app as rag_app

        _rag_engine = answer_question_graph
        _rag_app = rag_app
        print("[INFO] RAG engine loaded successfully")
    return _rag_engine, _rag_app


# ------------------------------------------------------------------
# OpenAI 호환 메시지 포맷
# ------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = "rag-gpt"
    messages: List[Message]
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


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


# ------------------------------------------------------------------
# 토큰 카운트
# ------------------------------------------------------------------
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text or ""))
    except Exception:
        return len((text or "").split())


# ------------------------------------------------------------------
# 스트리밍 답변 생성기
# ------------------------------------------------------------------
async def stream_rag_answer(question: str) -> AsyncGenerator[str, None]:
    """RAG 답변을 토큰 단위로 스트리밍"""
    try:
        # RAG 엔진 지연 로딩
        answer_func, rag_app = get_rag_engine()

        # RAG 그래프 실행하여 전체 답변 생성
        result = rag_app.invoke({"question": question})
        answer = result.get("answer", "죄송합니다. 답변을 생성할 수 없습니다.")

        # 답변을 토큰 단위로 분할하여 스트리밍
        words = answer.split()
        for i, word in enumerate(words):
            # 마지막 단어가 아니면 공백 추가
            if i < len(words) - 1:
                yield word + " "
            else:
                yield word

            # 스트리밍 효과를 위한 약간의 지연
            await asyncio.sleep(0.02)  # 20ms 지연

    except Exception as e:
        error_msg = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        words = error_msg.split()
        for i, word in enumerate(words):
            if i < len(words) - 1:
                yield word + " "
            else:
                yield word
            await asyncio.sleep(0.02)


# ------------------------------------------------------------------
# OpenAI Chat Completions (비스트리밍 + 스트리밍)
# ------------------------------------------------------------------
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest, authorization: Optional[str] = Header(None)
):
    try:
        user_messages = [m.content for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found.")
        question = user_messages[-1]

        # 스트리밍 처리
        if request.stream:

            async def sse_generator():
                created_ts = int(time.time())
                resp_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                model_id = request.model or "rag-gpt"

                # 1. role delta 전송
                chunk_role = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_role, ensure_ascii=False)}\n\n"

                # 2. 토큰 단위로 content 스트리밍
                async for token in stream_rag_answer(question):
                    chunk_content = {
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_content, ensure_ascii=False)}\n\n"

                # 3. 종료 신호
                chunk_finish = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_finish, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
                },
            )

        # 비스트리밍 처리
        answer_func, _ = get_rag_engine()
        answer = answer_func(question) or "죄송합니다. 답변을 생성할 수 없습니다."

        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created_timestamp = int(time.time())

        prompt_tokens = count_tokens(question, request.model or "gpt-4o")
        completion_tokens = count_tokens(answer, request.model or "gpt-4o")
        total_tokens = prompt_tokens + completion_tokens

        response = ChatResponse(
            id=response_id,
            object="chat.completion",
            created=created_timestamp,
            model=request.model or "rag-gpt",
            choices=[
                ChatResponseChoice(
                    index=0,
                    message=ChatResponseMessage(role="assistant", content=answer),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ------------------------------------------------------------------
# 간단한 질문-답변 엔드포인트 (테스트용) - 스트리밍 지원 추가
# ------------------------------------------------------------------
@app.post("/api/ask")
async def simple_ask(question: dict):
    try:
        if "question" not in question:
            raise HTTPException(status_code=400, detail="Question field required")

        # 스트리밍 요청 확인
        if question.get("stream", False):

            async def simple_sse_generator():
                async for token in stream_rag_answer(question["question"]):
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                simple_sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # 비스트리밍
        answer_func, _ = get_rag_engine()
        answer = answer_func(question["question"])
        return {"question": question["question"], "answer": answer, "status": "success"}

    except Exception as e:
        print(f"Simple API 오류: {e}")
        return {
            "question": question.get("question", ""),
            "answer": "답변 생성 중 오류가 발생했습니다.",
            "status": "error",
            "error": str(e),
        }


# ------------------------------------------------------------------
# 헬스체크 & 모델 목록
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {"msg": "RAG API is running.", "status": "healthy"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-gpt",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rag-api",
            }
        ],
    }


# ------------------------------------------------------------------
# 상태 확인 엔드포인트 추가
# ------------------------------------------------------------------
@app.get("/api/status")
def get_status():
    """RAG 엔진 및 리랭커 상태 확인"""
    try:
        if _rag_engine is None:
            return {"rag_loaded": False, "reranker_status": "not_loaded"}

        # RAG 엔진이 로드된 경우 리랭커 상태 확인
        from rag_engine_v3 import get_reranker_status

        reranker_status = get_reranker_status()

        return {
            "rag_loaded": True,
            "reranker_status": reranker_status,
            "status": "ready",
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


if __name__ == "__main__":
    import uvicorn

    # reload=False로 설정하여 중복 로드 방지
    uvicorn.run(
        "rag_api_server:app",
        host="0.0.0.0",
        port=8910,
        reload=False,  # 개발 중이 아니라면 False로 설정
        workers=1,  # 단일 워커로 제한
    )
