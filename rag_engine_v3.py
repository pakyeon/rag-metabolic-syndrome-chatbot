from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TypedDict
import threading

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from vector_db_build_v5 import VectorDBBuilder  # 인덱스 없을 때 빌드용

# ----------------------
# Config & clients
# ----------------------
load_dotenv()

LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.2"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nlpai-lab/KURE-v1")
CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", f"./chromadb/v2/{EMBED_MODEL}")
os.makedirs(CHROMA_DIR, exist_ok=True)

llm = ChatOpenAI(model_name=LLM_MODEL, temperature=LLM_TEMPERATURE)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

detect_llm = ChatOpenAI(model_name="gpt-5-nano", temperature=0)


# ----------------------
# Optional reranker with lazy loading and singleton pattern
# ----------------------
def _env_truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


RAG_USE_RERANK = _env_truthy(os.getenv("RAG_USE_RERANK", "0"))


class RerankerSingleton:
    """싱글톤 패턴으로 리랭커 모델 중복 로드 방지"""

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.tokenizer = None
                    self.model = None
                    self.token_false = None
                    self.token_true = None
                    self.prefix_tokens = None
                    self.suffix_tokens = None
                    self.max_len = 8192
                    self._loading = False
                    RerankerSingleton._initialized = True

    def _choose_torch_dtype(self):
        """최적의 torch dtype 선택"""
        if torch.cuda.is_available():
            # bf16 지원 시 우선
            if (
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            ):
                return torch.bfloat16
            # GPU는 fp16 일반적으로 OK
            return torch.float16
        # CPU는 fp32
        return torch.float32

    def ensure_loaded(self):
        """리랭커 모델을 지연 로딩 (thread-safe)"""
        if self.model is not None and self.tokenizer is not None:
            return True

        if self._loading:
            print(
                "[INFO] Reranker is already being loaded by another thread, waiting..."
            )
            # 다른 스레드에서 로딩 중일 때 대기
            while self._loading:
                threading.Event().wait(0.1)
            return self.model is not None and self.tokenizer is not None

        with self._lock:
            # Double-check locking pattern
            if self.model is not None and self.tokenizer is not None:
                return True

            if self._loading:
                return False

            self._loading = True

        try:
            print("[INFO] Loading Qwen3-Reranker-0.6B model (first use)...")

            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(
                    f"[INFO] GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                )

            dtype = self._choose_torch_dtype()
            model_name = "Qwen/Qwen3-Reranker-0.6B"

            # Tokenizer 로드 (dtype 불필요)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="left",
            )

            # Model 로드 with specific settings to prevent duplicate loading
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,  # 메모리 사용량 최적화
                trust_remote_code=False,  # 보안상 False
            ).eval()

            # 토큰 ID 미리 계산
            self.token_false = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true = self.tokenizer.convert_tokens_to_ids("yes")

            # 프롬프트 템플릿 토큰화
            prefix_text = (
                "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
                'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            )
            suffix_text = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
            self.prefix_tokens = self.tokenizer(prefix_text, add_special_tokens=False)[
                "input_ids"
            ]
            self.suffix_tokens = self.tokenizer(suffix_text, add_special_tokens=False)[
                "input_ids"
            ]

            if torch.cuda.is_available():
                print(
                    f"[INFO] GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                )

            print(f"[INFO] Reranker loaded successfully on device: {self.model.device}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load reranker: {e}")
            self.tokenizer = None
            self.model = None
            return False
        finally:
            self._loading = False

    def is_loaded(self):
        """리랭커가 로드되어 있는지 확인"""
        return self.model is not None and self.tokenizer is not None


# 전역 리랭커 인스턴스
reranker = RerankerSingleton()


# ----------------------
# Vector DB helpers
# ----------------------
def load_or_build_db():
    try:
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        got = db.get(include=[])
        if not got.get("ids"):
            raise ValueError("VectorDB is empty.")
        print("[INFO] VectorDB ready")
        return db
    except Exception as e:
        print(f"[WARN] VectorDB load failed: {e}")
        print("[INFO] Building VectorDB.")
        try:
            builder = VectorDBBuilder(
                embedding_model_name=EMBED_MODEL,
                chromadb_path=CHROMA_DIR,
                min_content_length=30,
                min_chunk_size=100,
                max_merge_size=1000,
            )
            ok = builder.build_from_folder(
                folder_path="./datasets/v4", chunk_size=500, chunk_overlap=50
            )
            if not ok:
                raise RuntimeError("VectorDB build failed")
            return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        except Exception as build_err:
            print("[ERROR] VectorDB build error:", build_err)
            traceback.print_exc()
            sys.exit(1)


DB = load_or_build_db()

RETRIEVER = DB.as_retriever(search_type="similarity", search_kwargs={"k": 20})

TOP_K = int(os.getenv("RAG_TOP_K", "20"))


# ----------------------
# State
# ----------------------
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


# ----------------------
# Nodes
# ----------------------
def n_classify(state: RAGState) -> RAGState:
    q = state["question"]
    prompt = ChatPromptTemplate.from_template(
        "질문이 '대사증후군'과 관련 있는지 'yes' 또는 'no'로만 답하시오.\n질문: {q}"
    )
    resp = (prompt | detect_llm).invoke({"q": q})
    related = resp.content.strip().lower().startswith("y")
    return {"is_related": related}


def n_retrieve(state: RAGState) -> RAGState:
    q = state["question"]
    try:
        results = DB.similarity_search_with_relevance_scores(q, k=TOP_K)
        if results:
            docs, scores = zip(*results)
            return {"raw_docs": list(docs), "scores": [float(s) for s in scores]}
        return {"raw_docs": [], "scores": []}
    except Exception:
        try:
            results = DB.similarity_search_with_score(q, k=TOP_K)  # (doc, distance)
            if results:
                docs, dists = zip(*results)
                scores = [1.0 - float(d) for d in dists]
                return {"raw_docs": list(docs), "scores": scores}
            return {"raw_docs": [], "scores": []}
        except Exception:
            docs = DB.similarity_search(q, k=TOP_K)
            return {"raw_docs": docs, "scores": []}


def n_score(state: RAGState) -> RAGState:
    docs = state.get("raw_docs", [])
    scores = state.get("scores", [])
    if not docs:
        return {"scored": []}
    scored = []
    for d, s in zip(docs, scores) if scores else [(d, None) for d in docs]:
        scored.append(
            {
                "document": d,
                "retrieval_score": (float(s) if s is not None else None),
                "source": d.metadata.get("source", "unknown"),
                "content": d.page_content,
            }
        )
    return {"scored": scored}


def _format_instruction(instruction: Optional[str], query: str, doc: str) -> str:
    if instruction is None:
        instruction = (
            "Given a query, judge whether the Document helps answer the query."
        )
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def _rerank_with_lazy_loading(
    query: str, doc_list: List[dict], top_k: int = 5
) -> List[dict]:
    """지연 로딩을 사용한 리랭킹"""
    if not RAG_USE_RERANK or not doc_list:
        return doc_list[:top_k]

    # 첫 사용 시에만 모델 로드
    if not reranker.ensure_loaded():
        print("[WARN] Reranker loading failed, using retrieval order")
        return doc_list[:top_k]

    try:
        pairs = [
            _format_instruction(None, query, d["document"].page_content)
            for d in doc_list
        ]

        # 토큰 길이 계산
        avail = (
            reranker.max_len - len(reranker.prefix_tokens) - len(reranker.suffix_tokens)
        )
        max_len = max(
            512,
            min(
                getattr(reranker.tokenizer, "model_max_length", reranker.max_len), avail
            ),
        )

        inputs = reranker.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            max_length=max_len,
        )

        # 프롬프트 템플릿 적용
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = (
                reranker.prefix_tokens + ids + reranker.suffix_tokens
            )

        inputs = reranker.tokenizer.pad(inputs, padding=True, return_tensors="pt").to(
            reranker.model.device
        )

        with torch.no_grad():
            logits = reranker.model(**inputs).logits[:, -1, :]
            true_logit = logits[:, reranker.token_true]
            false_logit = logits[:, reranker.token_false]
            scores = torch.softmax(
                torch.stack([false_logit, true_logit], dim=1), dim=1
            )[:, 1]

        reranked = [
            {**doc, "rerank_score": float(s)}
            for doc, s in zip(doc_list, scores.tolist())
        ]
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    except Exception as e:
        print(f"[WARN] Rerank failed, fallback to retrieval order. Reason: {e}")
        return doc_list[:top_k]


def n_rerank(state: RAGState) -> RAGState:
    scored = state.get("scored", [])
    if not scored:
        return {"reranked": []}
    if not RAG_USE_RERANK:
        k = min(5, len(scored))
        return {"reranked": scored[:k]}

    # 지연 로딩 리랭킹 사용
    rr = _rerank_with_lazy_loading(state["question"], scored, top_k=min(5, len(scored)))
    return {"reranked": rr}


def n_build_context(state: RAGState) -> RAGState:
    items = state.get("reranked") or state.get("scored") or []
    lines = []
    for i, d in enumerate(items, 1):
        md = getattr(d.get("document"), "metadata", {}) or {}
        label = md.get("header_path") or ""
        lines.append(f"{i}. 경로: {label}\n   내용: {d.get('content','')}\n")
    return {"context": "\n\n".join(lines)}


def n_generate_rag(state: RAGState) -> RAGState:
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "당신은 주어진 질문에 대해 핵심 내용을 간결하고 정확하게 답변하는 AI입니다.",
            ),
            (
                "user",
                "검색된 문서를 참고하여 답변하세요.\n\n질문:\n{question}\n\n검색된 문서:\n{documents}\n",
            ),
        ]
    )
    resp = (prompt | llm).invoke(
        {"question": state["question"], "documents": state.get("context", "")}
    )
    return {"answer": resp.content}


def n_generate_direct(state: RAGState) -> RAGState:
    prompt = ChatPromptTemplate.from_template("질문: {q}\n위 질문에 대해 답변하시오.")
    resp = (prompt | llm).invoke({"q": state["question"]})
    return {"answer": resp.content}


def n_guard_end(state: RAGState) -> RAGState:
    if state.get("is_related") and not state.get("raw_docs"):
        return {"error": "No documents retrieved"}
    return {}


# ----------------------
# Graph assembly with router
# ----------------------
graph = StateGraph(RAGState)

graph.add_node("classify", n_classify)
graph.add_node("retrieve", n_retrieve)
graph.add_node("score", n_score)
graph.add_node("rerank", n_rerank)
graph.add_node("build_context", n_build_context)
graph.add_node("generate_rag", n_generate_rag)
graph.add_node("generate_direct", n_generate_direct)
graph.add_node("guard_end", n_guard_end)


def route_is_related(state: RAGState) -> bool:
    return state.get("is_related", False)


graph.add_conditional_edges(
    "classify",
    route_is_related,
    {True: "retrieve", False: "generate_direct"},
)

graph.add_edge("retrieve", "score")
graph.add_edge("score", "rerank")
graph.add_edge("rerank", "build_context")
graph.add_edge("build_context", "generate_rag")
graph.add_edge("generate_rag", "guard_end")
graph.add_edge("generate_direct", "guard_end")

graph.add_edge(START, "classify")
graph.add_edge("guard_end", END)

app = graph.compile()


# ----------------------
# PNG 시각화 저장
# ----------------------
def save_viz_png(app, out_dir="./_viz"):
    from langchain_core.runnables.graph_mermaid import MermaidDrawMethod

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    png_path = Path(out_dir) / f"rag_graph_{ts}.png"

    g = app.get_graph(xray=True)
    try:
        g.draw_mermaid_png(output_file_path=str(png_path))
    except Exception:
        try:
            g.draw_mermaid_png(
                output_file_path=str(png_path),
                draw_method=MermaidDrawMethod.PUPPETEER,
            )
        except Exception:
            g.draw_mermaid_png(
                output_file_path=str(png_path),
                draw_method=MermaidDrawMethod.GRAPHVIZ,
            )
    print(f"[VIZ] PNG saved: {png_path}")


# ----------------------
# Public API
# ----------------------
def answer_question_graph(question: str) -> str:
    final = app.invoke({"question": question})
    if final.get("answer"):
        return final["answer"]
    return (
        f"죄송합니다. 답변을 생성하지 못했습니다. 사유: {final.get('error', 'unknown')}"
    )


def get_reranker_status() -> dict:
    """리랭커 상태 확인용 함수"""
    return {
        "enabled": RAG_USE_RERANK,
        "loaded": reranker.is_loaded(),
        "model_device": (
            str(reranker.model.device) if reranker.is_loaded() else "not loaded"
        ),
        "gpu_memory_allocated": (
            f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"
            if torch.cuda.is_available()
            else "N/A"
        ),
    }


# ----------------------
# CLI entry
# ----------------------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--viz", action="store_true", help="그래프 시각화 파일 저장")
    p.add_argument("--question", type=str, default="대사증후군이란?")
    p.add_argument("--status", action="store_true", help="리랭커 상태 확인")

    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--rerank",
        action="store_true",
        help="Enable reranker (loads model on first use)",
    )
    g.add_argument(
        "--no-rerank", action="store_true", help="Disable reranker (default)"
    )

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # CLI overrides for reranker
    if getattr(args, "rerank", False):
        RAG_USE_RERANK = True
    if getattr(args, "no_rerank", False):
        RAG_USE_RERANK = False

    if args.status:
        status = get_reranker_status()
        print("=== Reranker Status ===")
        for k, v in status.items():
            print(f"{k}: {v}")
        sys.exit(0)

    do_viz = args.viz or os.getenv("RAG_VIZ", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if do_viz:
        save_viz_png(app, out_dir="./_viz")
    print(answer_question_graph(args.question))
