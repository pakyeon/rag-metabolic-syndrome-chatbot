import os
import re
import glob
import logging
from typing import List, Dict, Any, Optional

import random
import numpy as np
import torch


def set_global_seed(seed: int = 1) -> None:
    """시드 고정: 재현성 확보"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(1)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class VectorDBBuilder:
    def __init__(
        self,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        chromadb_path: str = "./chromadb",
        # 🔧 청크 분석/임계값 파라미터화
        min_content_length: int = 20,
        min_chunk_size: int = 50,
        max_merge_size: int = 800,
    ) -> None:
        """
        벡터 데이터베이스 구축 시스템 초기화

        Args:
            embedding_model_name: 사용할 임베딩 모델명
            chromadb_path: ChromaDB 저장 경로
            min_content_length: 헤더만 있는 청크 판정 시 내용 최소 길이
            min_chunk_size: 병합 대상 최소 청크 크기
            max_merge_size: 병합 결과 상한 크기
        """
        self.embedding_model_name = embedding_model_name
        self.chromadb_path = chromadb_path
        self.embeddings = None
        self.db: Optional[Chroma] = None

        # thresholds
        self.min_content_length = min_content_length
        self.min_chunk_size = min_chunk_size
        self.max_merge_size = max_merge_size

        # 임베딩 모델 초기화
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        """임베딩 모델 초기화"""
        try:
            # [0-2] 정규화 활성화: index 측과 query 측을 반드시 동일하게 맞춰야 함 :contentReference[oaicite:12]{index=12}
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(
                "임베딩 모델 초기화 완료: %s (normalize_embeddings=True)",
                self.embedding_model_name,
            )
        except Exception as e:  # pragma: no cover
            logger.error("임베딩 모델 초기화 실패: %s", str(e))
            raise

    def load_md_files_from_folder(self, folder_path: str) -> List[Document]:
        """
        폴더에서 마크다운 파일들을 로드하여 Document 객체 리스트로 반환
        """
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
            except Exception as e:  # pragma: no cover
                logger.error("파일 읽기 실패 (%s): %s", file_path, str(e))

        return documents

    def _is_header_only_chunk(self, content: str) -> bool:
        """
        청크가 헤더만 포함하고 있는지 확인
        """
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
        """메타데이터에서 최상위 헤더 레벨을 유추"""
        header_keys = sorted([k for k in metadata.keys() if k.startswith("Header")])
        if not header_keys:
            return 0
        levels = []
        for k in header_keys:
            try:
                lvl = int(re.findall(r"\d+", k)[0])
                levels.append(lvl)
            except Exception:
                continue
        return min(levels) if levels else 0

    def _can_merge_chunks(
        self, chunk1_metadata: Dict[str, Any], chunk2_metadata: Dict[str, Any]
    ) -> bool:
        """
        두 청크가 헤더 계층 구조상 병합 가능한지 확인
        (비즈니스 룰은 보수적으로 유지)
        """
        level1 = self._get_header_level(chunk1_metadata)
        level2 = self._get_header_level(chunk2_metadata)

        # 둘 다 헤더가 없으면 병합 가능
        if level1 == 0 and level2 == 0:
            return True

        # 같은 레벨이면 병합 가능
        if level1 == level2 and level1 > 0:
            return True

        # 첫 번째가 상위 헤더이고 두 번째가 그보다 하위 레벨이면 (예: 2 -> 3, 2 -> 4) 두 단계까지 허용
        if level1 > 0 and (level1 < level2 <= level1 + 2):
            return True

        return False

    def _dedup_overlap(self, left: str, right: str, max_check: int = 1000) -> str:
        """
        left(=현재까지 병합된 텍스트)의 '접미사'와
        right(=다음 청크)의 '접두사'가 겹치면 그 겹치는 길이만큼 right를 잘라 반환.
        - right 전체가 이미 left 안에 포함되어 있으면 빈 문자열을 반환(중복 삽입 방지).
        - max_check: 겹침 탐색 최대 길이(성능/안정성용)
        """
        if not left or not right:
            return right

        # 완전 포함: right 전체가 이미 포함된 경우
        if right.strip() and right in left:
            return ""

        max_len = min(len(left), len(right), max_check)

        # 긴 겹침부터 탐색
        for k in range(max_len, 0, -1):
            if left[-k:] == right[:k]:
                return right[k:]

        return right

    def _merge_small_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        작은 청크/헤더만 있는 청크를 다음 청크와 병합 (문서 경계 보존)
        """
        if not chunks:
            return chunks

        merged_chunks: List[Document] = []
        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            current_content = current_chunk.page_content.strip()
            current_source = current_chunk.metadata.get("source")

            # 충분히 크고 의미가 있으면 그대로 채택
            if len(
                current_content
            ) >= self.min_chunk_size and not self._is_header_only_chunk(
                current_content
            ):
                merged_chunks.append(current_chunk)
                i += 1
                continue

            # 병합 시작
            merged_content = current_content
            merged_metadata = dict(current_chunk.metadata)
            j = i + 1

            while j < len(chunks) and len(merged_content) < self.max_merge_size:
                next_chunk = chunks[j]

                # 다른 파일이면 병합 중단 (문서 경계 보호)
                if next_chunk.metadata.get("source") != current_source:
                    break

                if not self._can_merge_chunks(merged_metadata, next_chunk.metadata):
                    break

                next_content = next_chunk.page_content.strip()

                # 1) 겹치는 접미사/접두사 제거
                deduped_next = self._dedup_overlap(
                    merged_content,
                    next_content,
                    max_check=self.max_merge_size,
                )

                # 2) 완전 포함이면(추가할 내용 없음) 다음 청크로 진행
                if deduped_next == "":
                    j += 1
                    continue

                # 3) 실제 이어붙였을 때 길이 검사
                candidate = (
                    merged_content
                    + ("\n\n" if merged_content and deduped_next else "")
                    + deduped_next
                )
                if len(candidate) > self.max_merge_size:
                    break

                merged_content = candidate

                # 헤더 메타데이터 보수적 병합
                for k, v in next_chunk.metadata.items():
                    if k not in merged_metadata:
                        merged_metadata[k] = v

                # 목표 크기 도달 시 종료
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
        """[0-1] Header 1~3을 ' / '로 연결한 경로 문자열 생성 (정적 메서드로 수정)"""
        parts = [md.get("Header 1"), md.get("Header 2"), md.get("Header 3")]
        parts = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
        return " / ".join(parts)

    def split_documents_with_markdown_headers(
        self,
        documents: List[Document],
        chunk_size: int = 400,
        chunk_overlap: int = 40,
    ) -> List[Document]:
        # ✅ 헤더 레벨 1~3까지만 사용
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False,
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks: List[Document] = []
        for doc in documents:
            md_header_splits = markdown_splitter.split_text(doc.page_content)

            for split_doc in md_header_splits:
                # 원본 + 헤더 메타데이터 병합
                merged_md = {**doc.metadata, **split_doc.metadata}

                # ✅ [0-1] header_path를 인덱싱 시점에 생성·저장
                header_path = self._compose_header_path(merged_md)
                merged_md["header_path"] = header_path

                if len(split_doc.page_content) > chunk_size:
                    for chunk in text_splitter.split_text(split_doc.page_content):
                        all_chunks.append(
                            Document(page_content=chunk, metadata=merged_md)
                        )
                else:
                    all_chunks.append(
                        Document(
                            page_content=split_doc.page_content, metadata=merged_md
                        )
                    )

        # (선택) 작은 청크 병합 로직 유지
        merged_chunks = self._merge_small_chunks(all_chunks)
        return merged_chunks

    def remove_duplicates(self, chunks: List[Document]) -> List[Document]:
        """
        중복되는 조각들을 제거 (순서는 유지)
        - 중복 판단 키: (정규화된 내용, source)
        """
        logger.info("중복 제거 중...")
        unique_chunks: List[Document] = []
        seen: set = set()

        def _normalize_text(t: str) -> str:
            return re.sub(r"\s+", " ", t.strip())

        for doc in chunks:
            norm = _normalize_text(doc.page_content)
            key = (norm, doc.metadata.get("source"))
            if key not in seen:
                seen.add(key)
                unique_chunks.append(doc)

        logger.info("중복 제거 후 조각 수: %d", len(unique_chunks))
        return unique_chunks

    def create_vector_database(self, chunks: List[Document]) -> Chroma:
        """
        벡터 데이터베이스 생성 (기존 경로가 있으면 삭제 후 생성)
        """
        if os.path.exists(self.chromadb_path):
            import shutil

            shutil.rmtree(self.chromadb_path)
            logger.info("기존 벡터 데이터베이스 삭제")

        self.db = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=self.chromadb_path,
        )
        try:
            # 일부 버전에서 명시 persist가 안전
            self.db.persist()  # type: ignore[attr-defined]
        except Exception:
            pass

        logger.info("벡터 데이터베이스 저장 완료: %s", self.chromadb_path)
        return self.db

    def analyze_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
        """청크 통계 정보 제공"""
        if not chunks:
            return {"총 청크 수": 0}

        lengths = [len(doc.page_content) for doc in chunks]
        return {
            "총 청크 수": len(chunks),
            "최소 길이": min(lengths),
            "최대 길이": max(lengths),
            "평균 길이": sum(lengths) // len(lengths),
        }

    def build_from_folder(
        self, folder_path: str, chunk_size: int = 400, chunk_overlap: int = 40
    ) -> bool:
        """
        폴더로부터 전체 파이프라인 수행
        """
        try:
            documents = self.load_md_files_from_folder(folder_path)
            if not documents:
                logger.error("처리할 문서가 없습니다.")
                return False

            chunks = self.split_documents_with_markdown_headers(
                documents, chunk_size, chunk_overlap
            )

            analysis = self.analyze_chunks(chunks)
            logger.info("\n=== 청크 분석 결과 ===")
            for k, v in analysis.items():
                logger.info("%s: %s", k, v)

            unique_chunks = self.remove_duplicates(chunks)
            logger.info("중복 제거 후 최종 청크 수: %d", len(unique_chunks))

            self.create_vector_database(unique_chunks)
            return True
        except Exception as e:  # pragma: no cover
            logger.exception("벡터 데이터베이스 구축 중 오류: %s", str(e))
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """
        데이터베이스 정보 제공
        """
        try:
            if not self.db:
                return {"status": "데이터베이스가 구축되지 않음"}

            try:
                got = self.db.get(include=[])  # ids만
                count = len(got.get("ids", []))
            except Exception:
                count = None

            info = {
                "status": "구축 완료",
                "total_documents": count,
                "embedding_model": self.embedding_model_name,
                "storage_path": self.chromadb_path,
                "database_size": self._get_directory_size(self.chromadb_path),
            }
            return info
        except Exception as e:  # pragma: no cover
            return {"status": f"정보 수집 중 오류: {str(e)}"}

    def _get_directory_size(self, path: str) -> str:
        """디렉토리 크기 계산"""
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except Exception:
                    pass
        return f"{total_size / (1024*1024):.2f} MB"


def main() -> None:
    """메인 실행 함수"""
    # 설정
    documents_folder = "./data"  # 마크다운 파일들이 있는 폴더
    embedding_model = "nlpai-lab/KURE-v1"  # 임베딩 모델
    chromadb_path = f"./chromadb/v2/{embedding_model}"  # 벡터DB 저장 경로
    chunk_size = 500  # 텍스트 조각 크기
    chunk_overlap = 50  # 조각 간 겹침

    load_dotenv()

    builder = VectorDBBuilder(
        embedding_model_name=embedding_model,
        chromadb_path=chromadb_path,
        # 🔧 필요 시 임계값 조정 가능
        min_content_length=30,
        min_chunk_size=100,
        max_merge_size=1000,
    )

    success = builder.build_from_folder(
        folder_path=documents_folder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if success:
        db_info = builder.get_database_info()
        logger.info("\n=== 데이터베이스 정보 ===")
        for key, value in db_info.items():
            logger.info("%s: %s", key, value)

        logger.info("\n벡터 데이터베이스 구축이 완료되었습니다!")
        logger.info("이제 retriever_evaluate.py를 사용하여 평가를 수행할 수 있습니다.")
    else:
        logger.error("벡터 데이터베이스 구축에 실패했습니다.")


if __name__ == "__main__":
    main()
