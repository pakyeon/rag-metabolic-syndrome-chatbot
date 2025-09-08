"""RAG 챗봇 통합 실행 스크립트"""

import argparse
import sys


def run_server(host="0.0.0.0", port=8910):
    """API 서버 실행"""
    import uvicorn

    uvicorn.run("src.api.server:app", host=host, port=port, reload=False, workers=1)


def build_db():
    """VectorDB 빌드"""
    # main() 함수를 호출하지 말고 직접 VectorDBBuilder를 사용
    from src.storage.vector_db import VectorDBBuilder
    from src.utils import set_global_seed
    from src import config

    # vector_db.py의 main() 함수와 동일한 로직을 직접 실행
    set_global_seed(1)

    builder = VectorDBBuilder(
        embedding_model_name=config.EMBED_MODEL,
        chromadb_path=config.CHROMA_DIR,
        min_content_length=config.MIN_CONTENT_LENGTH,
        min_chunk_size=config.MIN_CHUNK_SIZE,
        max_merge_size=config.MAX_MERGE_SIZE,
        raw_dir=config.RAW_DIR,
        parsed_dir=config.PARSED_DIR,
    )

    if builder.build(
        config.CHUNK_SIZE, config.CHUNK_OVERLAP, batch_size=config.DB_BATCH_SIZE
    ):
        print("\n=== DB 정보 ===")
        for k, v in builder.get_database_info().items():
            print(f"{k}: {v}")
        print("\n벡터 DB 빌드 완료!")
    else:
        print("벡터 DB 빌드 실패.")


def main():
    parser = argparse.ArgumentParser(description="RAG 챗봇 통합 실행 스크립트")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["server", "build-db"],
        default="server",
        help="실행할 명령 (기본값: server)",
    )
    parser.add_argument("--port", type=int, default=8910, help="서버 포트")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    args = parser.parse_args()

    if args.command == "server":
        run_server(args.host, args.port)
    elif args.command == "build-db":
        build_db()


if __name__ == "__main__":
    main()
