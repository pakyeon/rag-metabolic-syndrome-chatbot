import os
from dotenv import load_dotenv
from utils import _env_truthy

load_dotenv()

# --- RAG Engine Config ---
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.2"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nlpai-lab/KURE-v1")
CHROMA_DIR_BASE = os.getenv("RAG_CHROMA_DIR_BASE", "./chromadb/v2")
CHROMA_DIR = os.path.join(CHROMA_DIR_BASE, EMBED_MODEL)
RAG_USE_RERANK = _env_truthy(os.getenv("RAG_USE_RERANK", "0"))
TOP_K = int(os.getenv("RAG_TOP_K", "20"))
DETECT_LLM_MODEL = os.getenv("DETECT_LLM_MODEL", "gpt-5-nano")

# --- Redis Memory Config ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_TTL_HOURS = int(os.getenv("REDIS_TTL_HOURS", "1"))

# --- VectorDB Builder Config ---
DOCUMENTS_FOLDER = "./metabolic_syndrome_data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CONTENT_LENGTH = 30
MIN_CHUNK_SIZE = 100
MAX_MERGE_SIZE = 1000

# --- API Server Config ---
DEFAULT_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
ENV_ORIGINS = os.getenv("OPENWEBUI_ORIGINS", "")
ALLOW_ORIGINS = (
    [o.strip() for o in ENV_ORIGINS.split(",") if o.strip()]
    if ENV_ORIGINS
    else DEFAULT_ORIGINS
)
API_HOST = "0.0.0.0"
API_PORT = 8910
