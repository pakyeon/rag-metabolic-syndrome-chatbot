from __future__ import annotations
import os
import sys
import traceback
import argparse
from pathlib import Path
from datetime import datetime
from functools import partial

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END

from src import config
from src.core.reranker import reranker
from src.storage.vector_db import VectorDBBuilder
from src.core.graph_components import (
    RAGState,
    n_classify,
    n_retrieve,
    n_prepare_docs,
    n_rerank,
    n_build_context,
    n_generate_rag,
    n_generate_direct,
    n_guard_end,
)


def load_or_build_db():
    """벡터 DB를 로드하거나, 없으면 새로 빌드합니다."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        db = Chroma(persist_directory=config.CHROMA_DIR, embedding_function=embeddings)
        if not db._collection.count():
            raise ValueError("VectorDB is empty.")
        print("[INFO] VectorDB is ready.")
        return db
    except Exception as e:
        print(f"[WARN] VectorDB load failed: {e}. Building a new one.")
        try:
            builder = VectorDBBuilder(
                embedding_model_name=config.EMBED_MODEL,
                chromadb_path=config.CHROMA_DIR,
                min_content_length=config.MIN_CONTENT_LENGTH,
                min_chunk_size=config.MIN_CHUNK_SIZE,
                max_merge_size=config.MAX_MERGE_SIZE,
            )
            if not builder.build_from_folder(
                config.DOCUMENTS_FOLDER,
                config.CHUNK_SIZE,
                config.CHUNK_OVERLAP,
            ):
                raise RuntimeError("VectorDB build failed")

            embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBED_MODEL,
                encode_kwargs={"normalize_embeddings": True},
            )
            return Chroma(
                persist_directory=config.CHROMA_DIR, embedding_function=embeddings
            )
        except Exception as build_err:
            print(f"[ERROR] VectorDB build error: {build_err}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)


# --- Global resources ---
DB = load_or_build_db()


# --- Graph Assembly ---
def get_rag_app():
    """LangGraph RAG 애플리케이션을 구성하고 컴파일합니다."""
    graph = StateGraph(RAGState)

    graph.add_node("classify", n_classify)
    graph.add_node("retrieve", partial(n_retrieve, db=DB))
    graph.add_node("prepare_docs", n_prepare_docs)
    graph.add_node("rerank", n_rerank)
    graph.add_node("build_context", n_build_context)
    graph.add_node("generate_rag", n_generate_rag)
    graph.add_node("generate_direct", n_generate_direct)
    graph.add_node("guard_end", n_guard_end)

    graph.add_conditional_edges(
        "classify",
        lambda s: s.get("is_related", False),
        {True: "retrieve", False: "generate_direct"},
    )
    graph.add_edge("retrieve", "prepare_docs")
    graph.add_edge("prepare_docs", "rerank")
    graph.add_edge("rerank", "build_context")
    graph.add_edge("build_context", "generate_rag")
    graph.add_edge("generate_rag", "guard_end")
    graph.add_edge("generate_direct", "guard_end")
    graph.add_edge(START, "classify")
    graph.add_edge("guard_end", END)

    return graph.compile()


app = get_rag_app()


# --- Public API & Utilities ---
def answer_question_graph(question: str, conversation_history: str = None) -> str:
    """질문에 대한 답변을 생성합니다."""
    initial_state = {"question": question}
    if conversation_history:
        initial_state["conversation_history"] = conversation_history

    final_state = app.invoke(initial_state)
    return (
        final_state.get("answer")
        or f"죄송합니다. 답변을 생성하지 못했습니다. 사유: {final_state.get('error', 'unknown')}"
    )


def get_reranker_status() -> dict:
    """Reranker의 현재 상태를 반환합니다."""
    return {
        "enabled": config.RAG_USE_RERANK,
        "loaded": reranker.is_loaded(),
        "model_device": (
            str(reranker.model.device) if reranker.is_loaded() else "Not loaded"
        ),
        "gpu_memory": (
            f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"
            if torch.cuda.is_available()
            else "N/A"
        ),
    }


def save_viz_png(graph_app, out_dir="./_viz"):
    """그래프 시각화 이미지를 저장합니다."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    png_path = Path(out_dir) / f"rag_graph_{ts}.png"
    try:
        graph_app.get_graph(xray=True).draw_mermaid_png(output_file_path=str(png_path))
        print(f"[VIZ] Graph visualization saved to {png_path}")
    except Exception as e:
        print(f"[VIZ] Failed to save graph visualization: {e}")


# --- CLI ---
def main():
    """CLI 실행 함수"""
    parser = argparse.ArgumentParser(description="RAG Engine CLI")
    parser.add_argument(
        "-q", "--question", default="대사증후군이란?", help="테스트 질문"
    )
    parser.add_argument("--viz", action="store_true", help="그래프 시각화 파일 저장")
    parser.add_argument("--status", action="store_true", help="Reranker 상태 확인")
    parser.add_argument("--test-memory", action="store_true", help="메모리 기능 테스트")
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument(
        "--rerank", action="store_true", help="Reranker 사용 강제"
    )
    rerank_group.add_argument(
        "--no-rerank", action="store_true", help="Reranker 비사용 강제"
    )
    args = parser.parse_args()

    if args.rerank:
        config.RAG_USE_RERANK = True
    if args.no_rerank:
        config.RAG_USE_RERANK = False

    if args.status:
        print("=== Reranker Status ===")
        for k, v in get_reranker_status().items():
            print(f"{k}: {v}")
        return

    if args.test_memory:
        print("=== Memory Test ===")
        try:
            from src.storage.memory import memory

            # 테스트 메시지 추가
            memory.add_message("test_user", "test_conv", "user", "안녕하세요")
            memory.add_message(
                "test_user",
                "test_conv",
                "assistant",
                "안녕하세요! 무엇을 도와드릴까요?",
            )

            # 히스토리 조회
            history = memory.get_conversation_history("test_user", "test_conv")
            print(f"History: {history}")

            # 포맷된 히스토리
            formatted = memory.format_history_for_llm("test_user", "test_conv")
            print(f"Formatted: {formatted}")

            print("Memory test completed successfully!")
        except Exception as e:
            print(f"Memory test failed: {e}")
        return

    if args.viz:
        save_viz_png(app)

    print(f"\nQ: {args.question}")
    print("A:", answer_question_graph(args.question))


if __name__ == "__main__":
    main()
