import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ARTICLES_PATH = PROJECT_ROOT / "articles.json"
REVIEWS_PATH = PROJECT_ROOT / "reviews.json"
PROVIDERS_PATH = PROJECT_ROOT / "providers.json"

NORMALIZED_DIR = PROJECT_ROOT / "documents" / "normalized"
CHUNKED_DIR = PROJECT_ROOT / "documents" / "chunked"
INDEXED_DIR = PROJECT_ROOT / "documents" / "indexed"
GOLDEN_EVAL_PATH = PROJECT_ROOT / "evals" / "golden.json"

TARGET_MIN_TOKENS = 300
TARGET_MAX_TOKENS = 700
HARD_MAX_TOKENS = 1000
OVERLAP_TOKENS = 80

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")

EMBEDDER_PROVIDER = os.getenv("EMBEDDER", "openai")
GENERATOR_PROVIDER = os.getenv("GENERATOR", "openai")
OPENAI_EMBED_DIMENSIONS = 1536

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "getvoip_chunks")
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
RERANKER_PROVIDER = os.getenv("RERANKER", "cross-encoder")
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# Rough list prices for experiment cost tracking, not invoices.
OPENAI_EMBED_USD_PER_1M = float(os.getenv("OPENAI_EMBED_USD_PER_1M", "0.02"))
OPENAI_INPUT_USD_PER_1M = float(os.getenv("OPENAI_INPUT_USD_PER_1M", "0.40"))
OPENAI_OUTPUT_USD_PER_1M = float(os.getenv("OPENAI_OUTPUT_USD_PER_1M", "1.60"))
