import random
import numpy as np
import torch
import re
import os


def set_global_seed(seed: int = 1) -> None:
    """시드 고정: 재현성 확보"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_torch_dtype():
    """사용 가능한 최적의 torch dtype을 선택합니다."""
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """토큰 수를 계산합니다."""
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text or ""))
    except Exception:
        # tiktoken을 사용할 수 없을 경우, 공백 기준으로 단어 수를 계산합니다.
        return len((text or "").split())


def _env_truthy(v: str | None) -> bool:
    """환경 변수 값이 'true'를 의미하는지 확인합니다."""
    return (v or "").strip().lower() in ("1", "true", "TRUE")


def get_directory_size(path: str) -> str:
    """디렉토리 크기를 계산하여 문자열로 반환합니다."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                if not os.path.islink(filepath):
                    total_size += os.path.getsize(filepath)
            except Exception:
                pass
    return f"{total_size / (1024*1024):.2f} MB"
