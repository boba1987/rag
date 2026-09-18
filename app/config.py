import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ARTICLES_PATH = PROJECT_ROOT / "articles.json"
REVIEWS_PATH = PROJECT_ROOT / "reviews.json"
PROVIDERS_PATH = PROJECT_ROOT / "providers.json"

NORMALIZED_DIR = PROJECT_ROOT / "documents" / "normalized"
INDEXED_DIR = PROJECT_ROOT / "documents" / "indexed"
RAPTOR_DIR = PROJECT_ROOT / "documents" / "raptor"
GOLDEN_EVAL_PATH = PROJECT_ROOT / "evals" / "golden.json"
QUERY_DIR = PROJECT_ROOT / "documents" / "query"

CHUNKER_NAMES = ("structure_aware", "fixed_size", "parent_child")
ACTIVE_CHUNKER = os.getenv("CHUNKER", "structure_aware")

_CHUNKED_DIRS = {
    "structure_aware": PROJECT_ROOT / "documents" / "chunked",
    "fixed_size": PROJECT_ROOT / "documents" / "chunked_fixed",
    "parent_child": PROJECT_ROOT / "documents" / "chunked_parent_child",
}
_COLLECTION_ENV = {
    "structure_aware": "QDRANT_COLLECTION",
    "fixed_size": "QDRANT_COLLECTION_FIXED",
    "parent_child": "QDRANT_COLLECTION_PARENT_CHILD",
}
_COLLECTION_DEFAULTS = {
    "structure_aware": "getvoip_chunks_structure_aware",
    "fixed_size": "getvoip_chunks_fixed",
    "parent_child": "getvoip_chunks_parent_child",
}


def chunked_dir(chunker: str | None = None) -> Path:
    name = chunker or ACTIVE_CHUNKER
    if name not in _CHUNKED_DIRS:
        raise ValueError(f"Unknown chunker: {name}")
    return _CHUNKED_DIRS[name]


def qdrant_collection(chunker: str | None = None) -> str:
    name = chunker or ACTIVE_CHUNKER
    if name not in _COLLECTION_DEFAULTS:
        raise ValueError(f"Unknown chunker: {name}")
    return os.getenv(_COLLECTION_ENV[name], _COLLECTION_DEFAULTS[name])


CHUNKED_DIR = chunked_dir()
QDRANT_COLLECTIONS = {name: qdrant_collection(name) for name in CHUNKER_NAMES}

TARGET_MIN_TOKENS = 300
TARGET_MAX_TOKENS = 700
HARD_MAX_TOKENS = 1000
OVERLAP_TOKENS = 80

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")

EMBEDDER_PROVIDER = os.getenv("EMBEDDER", "openai")
GENERATOR_PROVIDER = os.getenv("GENERATOR", "openai")
QUERY_EXTRACTOR = os.getenv("QUERY_EXTRACTOR", "openai")
QUERY_EXTRACTOR_MODEL = os.getenv("QUERY_EXTRACTOR_MODEL", "gpt-4.1-nano")
RAPTOR_SUMMARIZER = os.getenv("RAPTOR_SUMMARIZER", "openai")
RAPTOR_SUMMARIZER_MODEL = os.getenv("RAPTOR_SUMMARIZER_MODEL", "gpt-4.1-nano")
RAPTOR_MAX_LEVELS = int(os.getenv("RAPTOR_MAX_LEVELS", "3"))
EVIDENCE_CHECKER = os.getenv("EVIDENCE_CHECKER", "heuristic")
EVIDENCE_MIN_OVERLAP = float(os.getenv("EVIDENCE_MIN_OVERLAP", "0.3"))
OPENAI_EMBED_DIMENSIONS = 1536

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = qdrant_collection()
QDRANT_COLLECTION_RAPTOR = os.getenv("QDRANT_COLLECTION_RAPTOR", "getvoip_chunks_raptor")
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
RERANKER_PROVIDER = os.getenv("RERANKER", "openai")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", QUERY_EXTRACTOR_MODEL)
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
BGE_RERANKER_MODEL = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-base")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "").lower()

# Rough list prices for experiment cost tracking, not invoices.
OPENAI_EMBED_USD_PER_1M = float(os.getenv("OPENAI_EMBED_USD_PER_1M", "0.02"))
OPENAI_INPUT_USD_PER_1M = float(os.getenv("OPENAI_INPUT_USD_PER_1M", "0.40"))
OPENAI_OUTPUT_USD_PER_1M = float(os.getenv("OPENAI_OUTPUT_USD_PER_1M", "1.60"))
