import os
import re
import glob
import logging
import argparse
import shutil
from typing import List, Dict, Any, Optional, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import config
from utils import set_global_seed, get_directory_size

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

PART_FILE_RE = re.compile(r"^part-(\d{2,3})\.md$", re.IGNORECASE)


class VectorDBBuilder:
    def __init__(
        self,
        embedding_model_name: str,
        chromadb_path: str,
        min_content_length: int,
        min_chunk_size: int,
        max_merge_size: int,
        raw_dir: str,
        parsed_dir: str,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.chromadb_path = chromadb_path
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.db: Optional[Chroma] = None
        self.min_content_length = min_content_length
        self.min_chunk_size = min_chunk_size
        self.max_merge_size = max_merge_size
        self.raw_dir = raw_dir
        self.parsed_dir = parsed_dir
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

    def _enumerate_md_parts(self) -> Dict[str, List[Tuple[int, str]]]:
        """
        parsed/<BASENAME>/part-XX.md 를 모두 수집하여
        {basename: [(part_index, md_path), ...]} 형태로 반환
        """
        result: Dict[str, List[Tuple[int, str]]] = {}
        if not os.path.isdir(self.parsed_dir):
            logger.warning("파싱 폴더가 없습니다: %s", self.parsed_dir)
            return result

        for basename in sorted(os.listdir(self.parsed_dir)):
            base_dir = os.path.join(self.parsed_dir, basename)
            if not os.path.isdir(base_dir):
                continue
            parts: List[Tuple[int, str]] = []
            for fname in sorted(os.listdir(base_dir)):
                m = PART_FILE_RE.match(fname)
                if not m:
                    continue
                idx = int(m.group(1))
                parts.append((idx, os.path.join(base_dir, fname)))
            if parts:
                parts.sort(key=lambda x: x[0])
                result[basename] = parts
        return result

    def _resolve_pdf_path(self, basename: str) -> Optional[str]:
        """
        raw/<BASENAME>.pdf 존재 확인 후 경로 반환.
        (없으면 None)
        """
        candidate = os.path.join(self.raw_dir, f"{basename}.pdf")
        return candidate if os.path.exists(candidate) else None

    def load_md_files(self) -> List[Document]:
        """
        parsed/<BASENAME>/part-XX.md를 로드하여 LangChain Document로 변환.
        메타데이터에 basename/part_index/part_count/pdf_path/md_path/source_id 포함.
        """
        documents: List[Document] = []
        mapping = self._enumerate_md_parts()
        if not mapping:
            logger.warning("파싱 문서를 찾지 못했습니다: %s", self.parsed_dir)
            return documents

        logger.info("발견된 문서 수(베이스네임): %d", len(mapping))
        for basename, parts in tqdm(
            mapping.items(), desc="Loading parsed docs", unit="doc"
        ):
            pdf_path = self._resolve_pdf_path(basename)
            if pdf_path is None:
                logger.warning("대응 PDF를 찾을 수 없습니다: raw/%s.pdf", basename)
            part_count = len(parts)
            for part_index, md_path in parts:
                try:
                    with open(md_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    meta = {
                        "source": pdf_path,  # 원본 문서
                        "basename": basename,  # 원본-파싱 공통 키
                        "part_index": part_index,  # 01부터 시작
                        "part_count": part_count,
                        "md_path": md_path if md_path else None,  # 파싱 문서
                        "source_id": f"{basename}#part-{part_index:02d}",
                    }
                    documents.append(Document(page_content=content, metadata=meta))
                except Exception as e:
                    logger.error("파일 읽기 실패 (%s): %s", md_path, str(e))
        logger.info("총 MD 파트 파일 수: %d", len(documents))
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
        with tqdm(total=len(chunks), desc="Merging small chunks", unit="chunk") as pbar:
            while i < len(chunks):
                current_chunk = chunks[i]
                current_content = current_chunk.page_content.strip()
                # 이미 충분히 길고 헤더만으로 구성되지 않았으면 그대로
                if len(
                    current_content
                ) >= self.min_chunk_size and not self._is_header_only_chunk(
                    current_content
                ):
                    merged_chunks.append(current_chunk)
                    i += 1
                    pbar.update(1)
                    continue

                merged_content = current_content
                merged_metadata = dict(current_chunk.metadata)
                j = i + 1
                # 같은 문서(source) 내에서만 병합
                while j < len(chunks) and len(merged_content) < self.max_merge_size:
                    next_chunk = chunks[j]
                    if next_chunk.metadata.get("source") != merged_metadata.get(
                        "source"
                    ):
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
                    # 보조 메타데이터 합치기(덮어쓰지 않음)
                    for k, v in next_chunk.metadata.items():
                        if k not in merged_metadata:
                            merged_metadata[k] = v

                    # 최소 크기 충족하면 stop 조건
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
                # 진행도는 소비한 청크 수만큼 올림
                pbar.update(max(1, j - i))
                i = j
        return merged_chunks

    def _compose_header_path(self, md: dict) -> str:
        """메타데이터로부터 헤더 경로 문자열을 생성합니다."""
        parts = [md.get(f"Header {i}") for i in range(1, 7)]
        return " > ".join(
            [p.strip() for p in parts if isinstance(p, str) and p.strip()]
        )

    def _extract_lower_level_headers(self, content: str) -> Dict[str, str]:
        """주어진 텍스트 블록에서 H4, H5, H6 헤더를 파싱하여 반환합니다."""
        headers = {}
        lines = content.split("\n")
        patterns = {
            "Header 4": re.compile(r"^\s*####\s+(.+)"),
            "Header 5": re.compile(r"^\s*#####\s+(.+)"),
            "Header 6": re.compile(r"^\s*######\s+(.+)"),
        }
        # 각 레벨별로 가장 먼저 나타나는 헤더 하나만 저장
        for key, pattern in patterns.items():
            for line in lines:
                match = pattern.match(line)
                if match:
                    headers[key] = match.group(1).strip()
                    break
        return headers

    def split_documents(
        self, documents: List[Document], chunk_size: int, chunk_overlap: int
    ) -> List[Document]:
        """문서를 청크로 분할합니다."""
        # 분할 기준은 H1~H3로 유지합니다.
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
        for doc in tqdm(documents, desc="Splitting docs", unit="doc"):
            # 1. splitter는 H1~H3을 기준으로 텍스트를 자르고 해당 메타데이터를 반환합니다.
            for split in splitter.split_text(doc.page_content):
                # 2. H1~H3 메타데이터를 기본으로 가져옵니다.
                merged_md = {**doc.metadata, **split.metadata}

                # 3. 잘린 청크의 내용에서 H4~H6 헤더를 직접 파싱하여 메타데이터에 추가합니다.
                lower_headers = self._extract_lower_level_headers(split.page_content)
                merged_md.update(lower_headers)

                # 4. H1~H6 정보가 모두 포함된 메타데이터로 전체 헤더 경로를 생성합니다.
                merged_md["header_path"] = self._compose_header_path(merged_md)

                # 5. 이후 로직은 동일하게, 크기에 따라 추가 분할을 진행합니다.
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
        for doc in tqdm(chunks, desc="Deduplicating", unit="chunk"):
            key = (normalize(doc.page_content), doc.metadata.get("source"))
            if key not in seen:
                seen.add(key)
                unique_chunks.append(doc)
        logger.info("중복 제거 전 %d개 -> 후 %d개", len(chunks), len(unique_chunks))
        return unique_chunks

    def create_vector_database(self, chunks: List[Document], batch_size: int) -> None:
        """벡터 데이터베이스를 생성하고 저장합니다."""
        if os.path.exists(self.chromadb_path):
            shutil.rmtree(self.chromadb_path)
            logger.info("기존 벡터 DB 삭제: %s", self.chromadb_path)

        # Chroma DB 초기화
        self.db = Chroma(
            persist_directory=self.chromadb_path,
            embedding_function=self.embeddings,
        )

        for i in tqdm(
            range(0, len(chunks), batch_size),
            desc="Adding documents to Chroma",
            unit="batch",
        ):
            batch = chunks[i : i + batch_size]
            self.db.add_documents(documents=batch)

        try:
            # 모든 문서 추가 후 persist를 호출합니다.
            self.db.persist()
        except Exception:
            pass
        logger.info("벡터 DB 저장 완료: %s", self.chromadb_path)

    def build(self, chunk_size: int, chunk_overlap: int, batch_size: int) -> bool:
        try:
            with logging_redirect_tqdm():
                docs = self.load_md_files()
                if not docs:
                    return False
                chunks = self.split_documents(docs, chunk_size, chunk_overlap)
                unique_chunks = self.remove_duplicates(chunks)
                logger.info("Chroma에 문서 쓰는 중…")
                self.create_vector_database(unique_chunks, batch_size=batch_size)
                logger.info("Chroma 저장 완료.")
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
                "status": "OK",
                "collection_count": self.db._collection.count(),  # type: ignore
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
        "--parsed-dir", default=config.PARSED_DIR, help="파싱 문서 루트 (parsed)"
    )
    parser.add_argument("--raw-dir", default=config.RAW_DIR, help="원본 PDF 루트 (raw)")
    parser.add_argument("--embed-model", default=config.EMBED_MODEL, help="임베딩 모델")
    parser.add_argument(
        "--db-path", default=config.CHROMA_DIR, help="ChromaDB 저장 경로"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=config.CHUNK_SIZE, help="청크 크기"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=config.CHUNK_OVERLAP, help="청크 겹침"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.DB_BATCH_SIZE,
        help="DB 추가 시 배치 크기",
    )
    args = parser.parse_args()

    builder = VectorDBBuilder(
        embedding_model_name=args.embed_model,
        chromadb_path=args.db_path,
        min_content_length=config.MIN_CONTENT_LENGTH,
        min_chunk_size=config.MIN_CHUNK_SIZE,
        max_merge_size=config.MAX_MERGE_SIZE,
        raw_dir=args.raw_dir,
        parsed_dir=args.parsed_dir,
    )
    if builder.build(args.chunk_size, args.chunk_overlap, batch_size=args.batch_size):
        logger.info("\n=== DB 정보 ===")
        for k, v in builder.get_database_info().items():
            logger.info("%s: %s", k, v)
        logger.info("\n벡터 DB 빌드 완료!")
    else:
        logger.error("벡터 DB 빌드 실패.")


if __name__ == "__main__":
    main()
