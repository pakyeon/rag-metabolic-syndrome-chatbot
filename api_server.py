from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional, AsyncGenerator
import asyncio
import re
import time
import uuid
import json
import hashlib

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
from redis_memory import memory

# ------------------------------------------------------------------------------
# RAG Engine - 서버 시작 시 로딩
# ------------------------------------------------------------------------------
print("[INFO] Loading RAG engine during server startup...")
try:
    from engine import (
        answer_question_graph,
        app as rag_app,
    )

    print("[INFO] RAG engine loaded successfully during startup")
except Exception as e:
    print(f"[ERROR] Failed to load RAG engine during startup: {e}")
    raise e

# ------------------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------------------
app = FastAPI(
    title="RAG API Server",
    description="FastAPI로 제공하는 RAG(검색증강생성) 질문-답변 서비스",
    version="0.4.0",
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
# RAG Engine getter (이제 이미 로딩된 것을 반환)
# ------------------------------------------------------------------------------
def get_rag_engine():
    """RAG 엔진을 반환 (서버 시작 시 이미 로딩됨)"""
    return answer_question_graph, rag_app


# ------------------------------------------------------------------------------
# 사용자/대화 ID 추출 유틸리티
# ------------------------------------------------------------------------------
def extract_user_conversation_id(
    request: ChatRequest, authorization: Optional[str] = None
) -> tuple[str, str]:
    """요청에서 사용자 ID와 대화 ID를 추출"""

    # 1. 요청 body에서 직접 추출
    user_id = request.user_id
    conversation_id = request.conversation_id

    # 2. Authorization 헤더에서 사용자 ID 추출 (간단한 방법)
    if not user_id and authorization:
        # "Bearer token" 형태에서 토큰 부분을 해시해서 사용자 ID로 사용
        token = authorization.replace("Bearer ", "").replace("bearer ", "")
        if token:
            user_id = hashlib.md5(token.encode()).hexdigest()[:12]

    # 3. 기본값 설정
    if not user_id:
        user_id = "default_user"

    if not conversation_id:
        conversation_id = "default_conversation"

    return user_id, conversation_id


# ------------------------------------------------------------------------------
# Streaming generator (server-sent events style)
# ------------------------------------------------------------------------------
async def stream_rag_answer(
    question: str, user_id: str = None, conversation_id: str = None
) -> AsyncGenerator[str, None]:
    """RAG 답변을 실제 토큰 단위로 스트리밍"""
    try:
        _, rag_app = get_rag_engine()

        # 대화 히스토리 조회
        conversation_history = ""
        if user_id and conversation_id:
            conversation_history = memory.format_history_for_llm(
                user_id, conversation_id
            )

        # 상태에 히스토리 정보 추가
        initial_state = {
            "question": question,
            "conversation_history": conversation_history,
            "user_id": user_id,
            "conversation_id": conversation_id,
        }

        async for message_chunk, metadata in rag_app.astream(
            initial_state, stream_mode="messages"
        ):
            # classify 노드는 제외하고 답변 생성 노드만 스트리밍
            node_name = metadata.get("langgraph_node", "")
            if node_name in ["generate_rag", "generate_direct"]:
                if hasattr(message_chunk, "content") and message_chunk.content:
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

        # 사용자/대화 ID 추출
        user_id, conversation_id = extract_user_conversation_id(request, authorization)

        # --- streaming ---
        if request.stream:

            async def sse_generator():
                created_ts = int(time.time())
                resp_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                model_id = request.model or "rag-gpt"

                # 사용자 메시지를 Redis에 저장
                memory.add_message(user_id, conversation_id, "user", question)

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
                full_answer = ""
                async for token in stream_rag_answer(
                    question, user_id, conversation_id
                ):
                    full_answer += token
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

                # 어시스턴트 답변을 Redis에 저장
                if full_answer.strip():
                    memory.add_message(
                        user_id, conversation_id, "assistant", full_answer.strip()
                    )

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
        # 사용자 메시지를 Redis에 저장
        memory.add_message(user_id, conversation_id, "user", question)

        # 대화 히스토리 조회
        conversation_history = memory.format_history_for_llm(user_id, conversation_id)

        # RAG 답변 생성 (히스토리 포함)
        _, rag_app = get_rag_engine()
        initial_state = {
            "question": question,
            "conversation_history": conversation_history,
            "user_id": user_id,
            "conversation_id": conversation_id,
        }
        final_state = rag_app.invoke(initial_state)
        answer = final_state.get("answer") or "죄송합니다. 답변을 생성할 수 없습니다."

        # 어시스턴트 답변을 Redis에 저장
        memory.add_message(user_id, conversation_id, "assistant", answer)

        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created_timestamp = int(time.time())

        prompt_tokens = count_tokens(question, request.model)
        completion_tokens = count_tokens(answer, request.model)
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
        # 사용자/대화 ID 설정
        user_id = body.user_id or "simple_user"
        conversation_id = body.conversation_id or "simple_conversation"

        if body.stream:

            async def simple_sse_generator():
                # 사용자 메시지를 Redis에 저장
                memory.add_message(user_id, conversation_id, "user", body.question)

                full_answer = ""
                async for token in stream_rag_answer(
                    body.question, user_id, conversation_id
                ):
                    full_answer += token
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

                # 어시스턴트 답변을 Redis에 저장
                if full_answer.strip():
                    memory.add_message(
                        user_id, conversation_id, "assistant", full_answer.strip()
                    )

                yield "data: [DONE]\n\n"

            return StreamingResponse(
                simple_sse_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # non-stream
        # 사용자 메시지를 Redis에 저장
        memory.add_message(user_id, conversation_id, "user", body.question)

        # 대화 히스토리 조회
        conversation_history = memory.format_history_for_llm(user_id, conversation_id)

        # RAG 답변 생성
        _, rag_app = get_rag_engine()
        initial_state = {
            "question": body.question,
            "conversation_history": conversation_history,
            "user_id": user_id,
            "conversation_id": conversation_id,
        }
        final_state = rag_app.invoke(initial_state)
        answer = final_state.get("answer") or "답변을 생성할 수 없습니다."

        # 어시스턴트 답변을 Redis에 저장
        memory.add_message(user_id, conversation_id, "assistant", answer)

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
# Memory Management APIs
# ------------------------------------------------------------------------------
@app.delete("/api/memory/{user_id}/{conversation_id}")
async def clear_conversation_memory(user_id: str, conversation_id: str):
    """특정 대화의 메모리 삭제"""
    try:
        success = memory.clear_conversation(user_id, conversation_id)
        return {
            "status": "success" if success else "failed",
            "message": f"Conversation memory cleared for {user_id}/{conversation_id}",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/memory/{user_id}/{conversation_id}")
async def get_conversation_memory(user_id: str, conversation_id: str):
    """특정 대화의 메모리 조회"""
    try:
        history = memory.get_conversation_history(user_id, conversation_id)
        return {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "history": history,
            "message_count": len(history),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


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
    """RAG 엔진 및 리랭커, Redis 상태 확인"""
    try:
        # RAG 엔진은 이미 로딩되어 있으므로 바로 상태를 확인할 수 있습니다
        from engine import get_reranker_status

        rag_status = {
            "rag_loaded": True,
            "reranker_status": get_reranker_status(),
        }

        # Redis 상태 확인
        redis_status = {"connected": False, "error": None}
        if memory.redis_client:
            try:
                memory.redis_client.ping()
                redis_status["connected"] = True
            except Exception as e:
                redis_status["error"] = str(e)

        return {
            **rag_status,
            "redis_memory": redis_status,
            "status": "ready",
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host=API_HOST, port=API_PORT, reload=False, workers=1)
