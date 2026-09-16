from qdrant_client import QdrantClient

from app.ingestion.indexer import upsert_chunks
from app.models.schemas import Chunk
from app.retrieval.dense import DenseRetriever

_PRICING = Chunk(
    id="provider_8019_pricing_01",
    document_id="8019",
    content_type="provider",
    provider="Nextiva",
    title="Nextiva",
    section="Pricing",
    heading_path=["Nextiva", "Pricing"],
    text="Core starts at $15 per user.",
    source_url="http://localhost/review/nextiva-2/",
    updated_at="2026-09-02 06:31:16",
)
_SUPPORT = Chunk(
    id="provider_8019_support_01",
    document_id="8019",
    content_type="provider",
    provider="Nextiva",
    title="Nextiva",
    section="Support",
    heading_path=["Nextiva", "Support"],
    text="24/7 phone and chat.",
    source_url="http://localhost/review/nextiva-2/",
    updated_at="2026-09-02 06:31:16",
)


class _FixedEmbedder:
    dimensions = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        mapping = {
            _PRICING.text: [1.0, 0.0, 0.0, 0.0],
            _SUPPORT.text: [0.0, 1.0, 0.0, 0.0],
            "How much does Nextiva cost?": [0.99, 0.01, 0.0, 0.0],
        }
        return [mapping[text] for text in texts]


def test_returns_closest_chunks_first() -> None:
    client = QdrantClient(":memory:")
    embedder = _FixedEmbedder()
    upsert_chunks(
        [_PRICING, _SUPPORT],
        embedder.embed([_PRICING.text, _SUPPORT.text]),
        embedder.dimensions,
        client=client,
        collection="dense_test",
    )
    retriever = DenseRetriever(
        embedder=embedder,
        client=client,
        collection="dense_test",
        top_k=2,
    )
    results = retriever.search("How much does Nextiva cost?")
    assert [chunk.section for chunk in results] == ["Pricing", "Support"]
    assert results[0].score > results[1].score
    assert results[0].text == _PRICING.text
    assert results[0].updated_at == _PRICING.updated_at


def test_respects_top_k() -> None:
    client = QdrantClient(":memory:")
    embedder = _FixedEmbedder()
    upsert_chunks(
        [_PRICING, _SUPPORT],
        embedder.embed([_PRICING.text, _SUPPORT.text]),
        embedder.dimensions,
        client=client,
        collection="dense_test",
    )
    retriever = DenseRetriever(
        embedder=embedder,
        client=client,
        collection="dense_test",
        top_k=5,
    )
    results = retriever.search("How much does Nextiva cost?", top_k=1)
    assert len(results) == 1
    assert results[0].section == "Pricing"
