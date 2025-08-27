import os
import re
import glob
import logging
import argparse
import shutil
from typing import List, Dict, Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils import set_global_seed, get_directory_size
import config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


class VectorDBBuilder:
    def __init__(
        self,
        embedding_model_name: str,
        chromadb_path: str,
        min_content_length: int,
        min_chunk_size: int,
        max_merge_size: int,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.chromadb_path = chromadb_path
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.db: Optional[Chroma] = None
        self.min_content_length = min_content_length
        self.min_chunk_size = min_chunk_size
        self.max_merge_size = max_merge_size
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        """임베딩 모델을 초기화합니다."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(
                "임베딩 모델 초기화 완료: %s (normalize_embeddings=True)",
                self.embedding_model_name,
            )
        except Exception as e:
            logger.error("임베딩 모델 초기화 실패: %s", str(e))
            raise

    def load_md_files_from_folder(self, folder_path: str) -> List[Document]:
        """폴더에서 마크다운 파일을 로드합니다."""
        documents: List[Document] = []
        if not os.path.exists(folder_path):
            logger.warning("폴더 '%s'가 존재하지 않습니다.", folder_path)
            return documents
        md_files = glob.glob(os.path.join(folder_path, "*.md"))
        if not md_files:
            logger.warning("폴더 '%s'에 .md 파일이 없습니다.", folder_path)
            return documents
        logger.info("발견된 .md 파일 수: %d", len(md_files))
        for file_path in md_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(
                        Document(page_content=content, metadata={"source": file_path})
                    )
            except Exception as e:
                logger.error("파일 읽기 실패 (%s): %s", file_path, str(e))
        return documents

    def _is_header_only_chunk(self, content: str) -> bool:
        """청크가 헤더만 포함하고 있는지 확인합니다."""
        header_pattern = r"^#{1,6}\s+.+$"
        lines = content.strip().split("\n")
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        if not non_empty_lines:
            return True
        header_only = all(re.match(header_pattern, line) for line in non_empty_lines)
        content_without_headers = "\n".join(
            [line for line in non_empty_lines if not re.match(header_pattern, line)]
        )
        too_short = len(content_without_headers.strip()) < self.min_content_length
        return header_only or too_short

    def _get_header_level(self, metadata: Dict[str, Any]) -> int:
        """메타데이터에서 헤더 레벨을 유추합니다."""
        header_keys = sorted([k for k in metadata.keys() if k.startswith("Header")])
        if not header_keys:
            return 0
        levels = [
            int(re.findall(r"\d+", k)[0]) for k in header_keys if re.findall(r"\d+", k)
        ]
        return min(levels) if levels else 0

    def _can_merge_chunks(self, meta1: Dict[str, Any], meta2: Dict[str, Any]) -> bool:
        """두 청크가 병합 가능한지 확인합니다."""
        level1, level2 = self._get_header_level(meta1), self._get_header_level(meta2)
        if level1 == 0 and level2 == 0:
            return True
        if level1 == level2 and level1 > 0:
            return True
        if level1 > 0 and (level1 < level2 <= level1 + 2):
            return True
        return False

    def _dedup_overlap(self, left: str, right: str) -> str:
        """두 텍스트의 중복 부분을 제거합니다."""
        if not left or not right:
            return right
        if right.strip() and right in left:
            return ""
        max_len = min(len(left), len(right), 1000)
        for k in range(max_len, 0, -1):
            if left.endswith(right[:k]):
                return right[k:]
        return right

    def _merge_small_chunks(self, chunks: List[Document]) -> List[Document]:
        """작은 청크들을 병합합니다."""
        if not chunks:
            return chunks
        merged_chunks: List[Document] = []
        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            current_content = current_chunk.page_content.strip()
            if len(
                current_content
            ) >= self.min_chunk_size and not self._is_header_only_chunk(
                current_content
            ):
                merged_chunks.append(current_chunk)
                i += 1
                continue

            merged_content = current_content
            merged_metadata = dict(current_chunk.metadata)
            j = i + 1
            while j < len(chunks) and len(merged_content) < self.max_merge_size:
                next_chunk = chunks[j]
                if next_chunk.metadata.get("source") != merged_metadata.get("source"):
                    break
                if not self._can_merge_chunks(merged_metadata, next_chunk.metadata):
                    break

                next_content = self._dedup_overlap(
                    merged_content, next_chunk.page_content.strip()
                )
                if not next_content:
                    j += 1
                    continue

                candidate = merged_content + "\n\n" + next_content
                if len(candidate) > self.max_merge_size:
                    break

                merged_content = candidate
                for k, v in next_chunk.metadata.items():
                    if k not in merged_metadata:
                        merged_metadata[k] = v

                if len(
                    merged_content
                ) >= self.min_chunk_size and not self._is_header_only_chunk(
                    merged_content
                ):
                    j += 1
                    break
                j += 1

            if merged_content.strip():
                merged_chunks.append(
                    Document(page_content=merged_content, metadata=merged_metadata)
                )
            i = j
        return merged_chunks

    @staticmethod
    def _compose_header_path(md: dict) -> str:
        """메타데이터로부터 헤더 경로 문자열을 생성합니다."""
        parts = [md.get(f"Header {i}") for i in range(1, 4)]
        return " / ".join(
            [p.strip() for p in parts if isinstance(p, str) and p.strip()]
        )

    def split_documents(
        self, documents: List[Document], chunk_size: int, chunk_overlap: int
    ) -> List[Document]:
        """문서를 청크로 분할합니다."""
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False,
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        all_chunks: List[Document] = []
        for doc in documents:
            for split in splitter.split_text(doc.page_content):
                merged_md = {**doc.metadata, **split.metadata}
                merged_md["header_path"] = self._compose_header_path(merged_md)
                if len(split.page_content) > chunk_size:
                    for chunk in text_splitter.split_text(split.page_content):
                        all_chunks.append(
                            Document(page_content=chunk, metadata=merged_md)
                        )
                else:
                    all_chunks.append(
                        Document(page_content=split.page_content, metadata=merged_md)
                    )

        return self._merge_small_chunks(all_chunks)

    def remove_duplicates(self, chunks: List[Document]) -> List[Document]:
        """중복 청크를 제거합니다."""
        unique_chunks: List[Document] = []
        seen = set()
        normalize = lambda t: re.sub(r"\s+", " ", t.strip())
        for doc in chunks:
            key = (normalize(doc.page_content), doc.metadata.get("source"))
            if key not in seen:
                seen.add(key)
                unique_chunks.append(doc)
        logger.info("중복 제거 전 %d개 -> 후 %d개", len(chunks), len(unique_chunks))
        return unique_chunks

    def create_vector_database(self, chunks: List[Document]) -> None:
        """벡터 데이터베이스를 생성하고 저장합니다."""
        if os.path.exists(self.chromadb_path):
            shutil.rmtree(self.chromadb_path)
            logger.info("기존 벡터 DB 삭제: %s", self.chromadb_path)
        self.db = Chroma.from_documents(
            chunks, self.embeddings, persist_directory=self.chromadb_path
        )
        try:
            # 일부 버전에서 명시 persist가 안전
            self.db.persist()
        except Exception:
            pass
        logger.info("벡터 DB 저장 완료: %s", self.chromadb_path)

    def build_from_folder(
        self, folder_path: str, chunk_size: int, chunk_overlap: int
    ) -> bool:
        """폴더로부터 전체 빌드 파이프라인을 실행합니다."""
        try:
            docs = self.load_md_files_from_folder(folder_path)
            if not docs:
                return False
            chunks = self.split_documents(docs, chunk_size, chunk_overlap)
            unique_chunks = self.remove_duplicates(chunks)
            self.create_vector_database(unique_chunks)
            return True
        except Exception as e:
            logger.exception("벡터 DB 빌드 중 오류 발생: %s", e)
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """생성된 데이터베이스의 정보를 반환합니다."""
        try:
            if not self.db:
                self.db = Chroma(
                    persist_directory=self.chromadb_path,
                    embedding_function=self.embeddings,
                )
            return {
                "status": "Ready",
                "total_documents": self.db._collection.count(),
                "embedding_model": self.embedding_model_name,
                "path": self.chromadb_path,
                "size": get_directory_size(self.chromadb_path),
            }
        except Exception as e:
            return {"status": f"Error: {e}"}


def main():
    """CLI 실행 함수"""
    set_global_seed(1)
    parser = argparse.ArgumentParser(description="벡터 DB 빌드 스크립트")
    parser.add_argument(
        "--docs-folder", default=config.DOCUMENTS_FOLDER, help="문서 폴더 경로"
    )
    parser.add_argument("--embed-model", default=config.EMBED_MODEL, help="임베딩 모델")
    parser.add_argument(
        "--db-path", default=None, help="ChromaDB 저장 경로 (기본값: rag_config 기반)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=config.CHUNK_SIZE, help="청크 크기"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=config.CHUNK_OVERLAP, help="청크 겹침"
    )
    args = parser.parse_args()

    db_path = args.db_path or os.path.join(config.CHROMA_DIR_BASE, args.embed_model)

    builder = VectorDBBuilder(
        embedding_model_name=args.embed_model,
        chromadb_path=db_path,
        min_content_length=config.MIN_CONTENT_LENGTH,
        min_chunk_size=config.MIN_CHUNK_SIZE,
        max_merge_size=config.MAX_MERGE_SIZE,
    )
    if builder.build_from_folder(args.docs_folder, args.chunk_size, args.chunk_overlap):
        logger.info("\n=== DB 정보 ===")
        for k, v in builder.get_database_info().items():
            logger.info("%s: %s", k, v)
        logger.info("\n벡터 DB 빌드 완료!")
    else:
        logger.error("벡터 DB 빌드 실패.")


if __name__ == "__main__":
    main()
