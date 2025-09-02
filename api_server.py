# rag_api_server.py (refactored full)
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional, AsyncGenerator
import asyncio
import re
import time
import uuid
import json

# === import shared models & utils ===
from api_models import (
    ChatRequest,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseMessage,
    Usage,
    Message,
    SimpleAskRequest,
)
from utils import count_tokens
from config import (
    ALLOW_ORIGINS as allow_origins,
    API_HOST,
    API_PORT,
)

# ------------------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------------------
app = FastAPI(
    title="RAG API Server",
    description="FastAPI로 제공하는 RAG(검색증강생성) 질문-답변 서비스",
    version="0.3.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Lazy-load RAG Engine
# ------------------------------------------------------------------------------
_rag_engine = None
_rag_app = None


def get_rag_engine():
    """RAG 엔진을 지연 로딩"""
    global _rag_engine, _rag_app
    if _rag_engine is None:
        print("[INFO] Loading RAG engine (first request)...")
        from engine import (
            answer_question_graph,
            app as rag_app,
        )

        _rag_engine = answer_question_graph
        _rag_app = rag_app
        print("[INFO] RAG engine loaded successfully")
    return _rag_engine, _rag_app


# ------------------------------------------------------------------------------
# Streaming generator (server-sent events style)
# ------------------------------------------------------------------------------
async def stream_rag_answer(question: str) -> AsyncGenerator[str, None]:
    """RAG 답변을 실제 토큰 단위로 스트리밍"""
    try:
        answer_func, rag_app = get_rag_engine()
        
        async for message_chunk, metadata in rag_app.astream(
            {"question": question},
            stream_mode="messages"
        ):
            # classify 노드는 제외하고 답변 생성 노드만 스트리밍
            node_name = metadata.get("langgraph_node", "")
            if node_name in ["generate_rag", "generate_direct"]:
                if hasattr(message_chunk, 'content') and message_chunk.content:
                    yield message_chunk.content

    except Exception as e:
        msg = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        for m in re.finditer(r"[^\s]+|\s+", msg):
            yield m.group(0)
            await asyncio.sleep(0.02)


# ------------------------------------------------------------------------------
# OpenAI Chat Completions (stream / non-stream)
# ------------------------------------------------------------------------------
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest, authorization: Optional[str] = Header(None)
):
    try:
        user_messages = [m.content for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found.")
        question = user_messages[-1]

        # --- streaming ---
        if request.stream:

            async def sse_generator():
                created_ts = int(time.time())
                resp_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                model_id = request.model or "rag-gpt"

                # 1) role delta
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

                # 2) token deltas
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

                # 3) finish
                chunk_finish = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk_finish, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # --- non-stream ---
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


# ------------------------------------------------------------------------------
# Simple Ask (테스트용, stream 지원)
# ------------------------------------------------------------------------------
@app.post("/api/ask")
async def simple_ask(body: SimpleAskRequest):
    try:
        if body.stream:

            async def simple_sse_generator():
                async for token in stream_rag_answer(body.question):
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                simple_sse_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        answer_func, _ = get_rag_engine()
        answer = answer_func(body.question)
        return {"question": body.question, "answer": answer, "status": "success"}

    except Exception as e:
        print(f"Simple API 오류: {e}")
        return {
            "question": body.question,
            "answer": "답변 생성 중 오류가 발생했습니다.",
            "status": "error",
            "error": str(e),
        }


# ------------------------------------------------------------------------------
# Health / Models / Status
# ------------------------------------------------------------------------------
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


@app.get("/api/status")
def get_status():
    """RAG 엔진 및 리랭커 상태 확인"""
    try:
        if _rag_engine is None:
            return {"rag_loaded": False, "reranker_status": "not_loaded"}

        from engine import (
            get_reranker_status,
        )

        return {
            "rag_loaded": True,
            "reranker_status": get_reranker_status(),
            "status": "ready",
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host=API_HOST, port=API_PORT, reload=False, workers=1)
