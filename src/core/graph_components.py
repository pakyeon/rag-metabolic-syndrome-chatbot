from __future__ import annotations
from typing import List, TypedDict, Optional, Dict, Any

import torch
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src import config
from src.core.reranker import reranker

# --- LLM Clients ---
llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
detect_llm = ChatOpenAI(model=config.DETECT_LLM_MODEL, temperature=0)


# --- State ---
class RAGState(TypedDict, total=False):
    question: str
    is_related: bool
    raw_docs: List[Document]
    scores: List[float]
    scored: List[dict]
    reranked: List[dict]
    context: str
    answer: str
    error: Optional[str]
    # 메모리 관련 필드 추가
    conversation_history: Optional[str]
    user_id: Optional[str]
    conversation_id: Optional[str]


# --- Nodes ---
def n_classify(state: RAGState) -> RAGState:
    """질문이 주제와 관련 있는지 분류합니다."""
    q = state["question"]
    prompt = ChatPromptTemplate.from_template(
        "질문이 '대사증후군'과 관련 있는지 'yes' 또는 'no'로만 답하시오.\n질문: {q}"
    )
    resp = (prompt | detect_llm).invoke({"q": q})
    related = resp.content.strip().lower().startswith("y")
    return {"is_related": related}


def n_retrieve(state: RAGState, db) -> RAGState:
    """DB에서 관련 문서를 검색합니다."""
    q = state["question"]
    try:
        results_with_scores = db.similarity_search_with_relevance_scores(
            q, k=config.TOP_K
        )
        if results_with_scores:
            docs, scores = zip(*results_with_scores)
            return {"raw_docs": list(docs), "scores": [float(s) for s in scores]}
        return {"raw_docs": [], "scores": []}
    except Exception:
        # relevance score를 지원하지 않는 일부 ChromaDB 버전을 위한 폴백
        docs = db.similarity_search(q, k=config.TOP_K)
        return {"raw_docs": docs, "scores": []}


def n_score(state: RAGState) -> RAGState:
    """검색된 문서에 점수를 매기고 형식을 지정합니다."""
    docs = state.get("raw_docs", [])
    scores = state.get("scores", [])
    if not docs:
        return {"scored": []}

    scored_docs = []
    for d, s in zip(docs, scores) if scores else [(d, None) for d in docs]:
        scored_docs.append(
            {
                "document": d,
                "retrieval_score": float(s) if s is not None else None,
                "source": d.metadata.get("source", "unknown"),
                "content": d.page_content,
            }
        )
    return {"scored": scored_docs}


def _format_rerank_instruction(query: str, doc: str) -> str:
    """Reranker를 위한 프롬프트 형식을 만듭니다."""
    instruction = "Given a query, judge whether the Document helps answer the query."
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def _rerank_docs(query: str, doc_list: List[dict], top_k: int = 5) -> List[dict]:
    """문서 목록을 리랭킹합니다."""
    if not reranker.ensure_loaded():
        print("[WARN] Reranker 로드 실패, 검색 순서를 사용합니다.")
        return doc_list[:top_k]

    try:
        pairs = [
            _format_rerank_instruction(query, d["document"].page_content)
            for d in doc_list
        ]

        # 토큰화 및 모델 입력 준비
        inputs = reranker.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            max_length=reranker.max_len
            - len(reranker.prefix_tokens)
            - len(reranker.suffix_tokens),
            return_attention_mask=False,
        )
        for i in range(len(inputs["input_ids"])):
            inputs["input_ids"][i] = (
                reranker.prefix_tokens + inputs["input_ids"][i] + reranker.suffix_tokens
            )

        padded_inputs = reranker.tokenizer.pad(
            inputs, padding=True, return_tensors="pt"
        ).to(reranker.model.device)

        # 점수 계산
        with torch.no_grad():
            logits = reranker.model(**padded_inputs).logits[:, -1, :]
            scores = torch.softmax(logits, dim=-1)[:, reranker.token_true]

        # 점수 추가 및 정렬
        for doc, score in zip(doc_list, scores.tolist()):
            doc["rerank_score"] = float(score)

        reranked_list = sorted(doc_list, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_list[:top_k]

    except Exception as e:
        print(f"[WARN] 리랭킹 실패, 검색 순서를 사용합니다. 이유: {e}")
        return doc_list[:top_k]


def n_rerank(state: RAGState) -> RAGState:
    """필요 시 문서를 리랭킹합니다."""
    scored = state.get("scored", [])
    if not scored:
        return {"reranked": []}
    if not config.RAG_USE_RERANK:
        return {"reranked": scored[:5]}

    reranked_docs = _rerank_docs(state["question"], scored, top_k=min(5, len(scored)))
    return {"reranked": reranked_docs}


def n_build_context(state: RAGState) -> RAGState:
    """LLM에 전달할 컨텍스트를 구성합니다."""
    items = state.get("reranked") or state.get("scored") or []
    lines = []
    for i, d in enumerate(items, 1):
        metadata = getattr(d.get("document"), "metadata", {}) or {}
        header_path = metadata.get("header_path", "")
        lines.append(f"{i}. 출처: {header_path}\n   내용: {d.get('content','')}\n")
    return {"context": "\n\n".join(lines)}


def n_generate_rag(state: RAGState) -> RAGState:
    """검색된 컨텍스트를 기반으로 답변을 생성합니다."""

    # 대화 히스토리 포함한 프롬프트 구성
    conversation_history = state.get("conversation_history", "")

    if conversation_history:
        system_message = """당신은 대사증후군 상담사를 지원하는 도우미입니다.
답변은 한국어로 작성하며, 가독성이 좋도록 문단과 줄바꿈, 목록 등을 적극적으로 활용하세요.
상담사가 환자에게 설명할 수 있도록 명확하고 이해하기 쉽게 작성하세요.

이전 대화 내용을 참고하여 문맥에 맞는 답변을 제공하세요."""

        user_message = """이전 대화:
{conversation_history}

현재 질문:
{question}

검색된 문서:
{documents}

위의 이전 대화 내용과 검색된 문서를 모두 참고하여 현재 질문에 답변하세요."""

    else:
        system_message = """당신은 대사증후군 상담사를 지원하는 도우미입니다.
답변은 한국어로 작성하며, 가독성이 좋도록 문단과 줄바꿈, 목록 등을 적극적으로 활용하세요.
상담사가 환자에게 설명할 수 있도록 명확하고 이해하기 쉽게 작성하세요."""

        user_message = "검색된 문서를 참고하여 답변하세요.\n\n질문:\n{question}\n\n검색된 문서:\n{documents}\n"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", user_message),
        ]
    )

    chain = prompt | llm

    invoke_params = {
        "question": state["question"],
        "documents": state.get("context", ""),
    }

    if conversation_history:
        invoke_params["conversation_history"] = conversation_history

    resp = chain.invoke(invoke_params)
    return {"answer": resp.content}


def n_generate_direct(state: RAGState) -> RAGState:
    """컨텍스트 없이 직접 답변을 생성합니다."""

    # 대화 히스토리 포함한 프롬프트 구성
    conversation_history = state.get("conversation_history", "")

    if conversation_history:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 AI 어시스턴트입니다.
이전 대화 내용을 참고하여 문맥에 맞는 답변을 제공하세요.
답변은 한국어로 작성하며, 명확하고 이해하기 쉽게 작성하세요.""",
                ),
                (
                    "user",
                    """이전 대화:
{conversation_history}

현재 질문:
{question}

위의 이전 대화 내용을 참고하여 현재 질문에 답변하세요.""",
                ),
            ]
        )

        resp = (prompt | llm).invoke(
            {
                "conversation_history": conversation_history,
                "question": state["question"],
            }
        )
    else:
        prompt = ChatPromptTemplate.from_template(
            "질문: {question}\n위 질문에 대해 답변하시오."
        )
        resp = (prompt | llm).invoke({"question": state["question"]})

    return {"answer": resp.content}


def n_guard_end(state: RAGState) -> RAGState:
    """그래프 종료 전 상태를 확인합니다."""
    if state.get("is_related") and not state.get("raw_docs"):
        return {"error": "관련 문서를 찾지 못했습니다."}
    return {}
