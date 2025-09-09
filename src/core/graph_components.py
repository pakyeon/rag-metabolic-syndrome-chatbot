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
    prepared: List[dict]
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


def n_prepare_docs(state: RAGState) -> RAGState:
    """
    [수정됨] 검색된 문서를 표준 형태로 정리합니다.
    (basename을 추가로 추출)
    """
    docs = state.get("raw_docs", [])
    scores = state.get("scores", [])
    if not docs:
        return {"prepared": []}

    prepared_docs = []
    for d, s in zip(docs, scores) if scores else [(d, None) for d in docs]:
        md = getattr(d, "metadata", {}) or {}
        prepared_docs.append(
            {
                "content": d.page_content,
                "basename": md.get("basename", "알 수 없는 문서"),
                "header_path": md.get("header_path", ""),
                "retrieval_score": float(s) if s is not None else None,
            }
        )
    return {"prepared": prepared_docs}


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
        pairs = [_format_rerank_instruction(query, d["content"]) for d in doc_list]

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
    items = state.get("reranked") or state.get("prepared") or []
    lines = []
    for i, d in enumerate(items, 1):
        basename = d.get("basename", "알 수 없는 문서")
        header_path = d.get("header_path") or "전체 내용"
        lines.append(
            f"{i}.\n"
            f"출처: {basename}\n"
            f"경로: {header_path}\n"
            f"내용: {d.get('content','')}\n"
        )
    return {"context": "\n\n".join(lines)}


def n_generate_rag(state: RAGState) -> RAGState:
    """개선된 RAG 기반 답변 생성 함수"""

    conversation_history = state.get("conversation_history", "")

    # 공통 시스템 메시지 (구조화 및 의료 면책사항 추가)
    system_message = """당신은 대사증후군 상담사를 지원하는 전문 AI 도우미입니다.

## 역할 및 책임
- 상담사가 환자에게 설명할 수 있도록 명확하고 이해하기 쉬운 정보 제공
- 검색된 문서의 내용을 바탕으로 정확하고 신뢰할 수 있는 답변 작성
- 한국어로 답변하며, 전문적이면서도 친근한 톤 유지

## 답변 형식 가이드라인
1. **구조화된 답변**: 주제별로 명확히 구분하여 작성
2. **가독성 최적화**: 문단, 줄바꿈, 번호/글머리 기호 적극 활용
3. **핵심 정보 강조**: 중요한 내용은 **굵게** 표시
4. **단계별 설명**: 복잡한 내용은 순서대로 단계별 설명
5. **실용적 조언**: 환자가 실제로 실천할 수 있는 구체적 방법 제시

## 의료 정보 관련 주의사항
⚠️ **중요**: 제공되는 정보는 일반적인 교육 목적이며, 개별 환자의 구체적인 의료 상담이나 진단을 대체할 수 없습니다. 환자별 맞춤 치료나 약물 조절은 반드시 담당 의료진과 상담하시기 바랍니다.

## 출처 표기 규칙
답변 마지막에는 반드시 다음 형식으로 참고 자료를 명시해야 합니다:

📚 **참고 자료**
- 출처: '[문서명]'
  경로: '[상세 경로]'
"""

    if conversation_history:
        user_message = """## 이전 대화 내용
{conversation_history}

## 현재 질문
{question}

## 검색된 관련 문서
{documents}

위의 이전 대화 맥락과 검색된 문서를 모두 참고하여, 현재 질문에 대한 체계적이고 실용적인 답변을 작성해주세요."""
    else:
        user_message = """## 질문
{question}

## 검색된 관련 문서
{documents}

위의 검색된 문서 내용을 바탕으로 질문에 대한 체계적이고 실용적인 답변을 작성해주세요."""

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
    """개선된 직접 답변 생성 함수"""

    conversation_history = state.get("conversation_history", "")

    # 공통 시스템 메시지
    system_message = """당신은 전문적이고 신뢰할 수 있는 AI 어시스턴트입니다.

## 답변 원칙
- 한국어로 명확하고 이해하기 쉽게 답변
- 구조화된 형식으로 정보 제공
- 전문적이면서도 친근한 톤 유지

## 답변 형식 가이드라인
1. **명확한 구조**: 주제별로 체계적 구성
2. **가독성**: 문단, 목록, 강조 표시 활용
3. **실용성**: 구체적이고 실행 가능한 정보 제공

## 의료/건강 관련 정보 주의사항
⚠️ 건강 관련 질문의 경우, 상담을 보조하는 참고 정보만 제공하며 개별적인 의료 상담은 전문 의료진과 상담하도록 안내합니다."""

    if conversation_history:
        user_message = """## 이전 대화 내용
{conversation_history}

## 현재 질문
{question}

이전 대화 맥락을 고려하여 현재 질문에 대한 적절한 답변을 제공해주세요."""
    else:
        user_message = """## 질문
{question}

위 질문에 대해 체계적이고 도움이 되는 답변을 제공해주세요."""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_message), ("user", user_message)]
    )

    invoke_params = {"question": state["question"]}

    if conversation_history:
        invoke_params["conversation_history"] = conversation_history

    resp = (prompt | llm).invoke(invoke_params)
    return {"answer": resp.content}


def n_guard_end(state: RAGState) -> RAGState:
    """그래프 종료 전 상태를 확인합니다."""
    if state.get("is_related") and not state.get("raw_docs"):
        return {"error": "관련 문서를 찾지 못했습니다."}
    return {}
