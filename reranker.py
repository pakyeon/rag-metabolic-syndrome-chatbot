import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer


class RerankerSingleton:
    """싱글톤 패턴으로 리랭커 모델 중복 로드를 방지합니다."""

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
        """사용 가능한 최적의 torch dtype을 선택합니다."""
        if torch.cuda.is_available():
            if (
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            ):
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def ensure_loaded(self):
        """리랭커 모델을 지연 로딩합니다 (Thread-safe)."""
        if self.model is not None and self.tokenizer is not None:
            return True

        if self._loading:
            print("[INFO] Reranker가 다른 스레드에서 로딩 중입니다. 잠시 대기합니다...")
            while self._loading:
                threading.Event().wait(0.1)
            return self.model is not None and self.tokenizer is not None

        with self._lock:
            if self.model is not None and self.tokenizer is not None:
                return True
            if self._loading:
                return False
            self._loading = True

        try:
            print("[INFO] Qwen/Qwen3-Reranker-0.6B 모델을 로드합니다 (최초 사용)...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(
                    f"[INFO] 모델 로드 전 GPU 메모리: {torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                )

            dtype = self._choose_torch_dtype()
            model_name = "Qwen/Qwen3-Reranker-0.6B"

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            ).eval()

            self.token_false = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true = self.tokenizer.convert_tokens_to_ids("yes")

            prefix_text = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            suffix_text = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
            self.prefix_tokens = self.tokenizer(prefix_text, add_special_tokens=False)[
                "input_ids"
            ]
            self.suffix_tokens = self.tokenizer(suffix_text, add_special_tokens=False)[
                "input_ids"
            ]

            if torch.cuda.is_available():
                print(
                    f"[INFO] 모델 로드 후 GPU 메모리: {torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                )
            print(f"[INFO] Reranker 로드 완료. 사용 디바이스: {self.model.device}")
            return True
        except Exception as e:
            print(f"[ERROR] Reranker 로드 실패: {e}")
            self.tokenizer = None
            self.model = None
            return False
        finally:
            self._loading = False

    def is_loaded(self):
        """리랭커 모델이 로드되었는지 확인합니다."""
        return self.model is not None and self.tokenizer is not None


# 전역 리랭커 인스턴스
reranker = RerankerSingleton()
