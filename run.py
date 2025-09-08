"""RAG 챗봇 통합 실행 스크립트"""

import argparse
import sys


def run_server(host="0.0.0.0", port=8910):
    """API 서버 실행"""
    import uvicorn

    uvicorn.run("src.api.server:app", host=host, port=port, reload=False, workers=1)


def build_db():
    """VectorDB 빌드"""
    from src.storage.vector_db import main

    main()


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
