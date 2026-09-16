from qdrant_client import QdrantClient

from app.ingestion.indexer import upsert_chunks
from app.models.schemas import Chunk, RetrievalFilters
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
_RC_PRICING = Chunk(
    id="provider_8015_pricing_01",
    document_id="8015",
    content_type="provider",
    provider="RingCentral",
    title="RingCentral",
    section="RingCentral Plans and Pricing",
    heading_path=["RingCentral", "RingCentral Plans and Pricing"],
    text="RingEX starts at $20 per user.",
    source_url="http://localhost/review/ringcentral-2/",
    updated_at="2026-04-13 10:03:33",
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
            _RC_PRICING.text: [0.8, 0.2, 0.0, 0.0],
            "How much does Nextiva cost?": [0.99, 0.01, 0.0, 0.0],
            "How much does RingCentral cost?": [0.8, 0.2, 0.0, 0.0],
        }
        return [mapping[text] for text in texts]


def _index(client: QdrantClient, embedder: _FixedEmbedder) -> None:
    chunks = [_PRICING, _SUPPORT, _RC_PRICING]
    upsert_chunks(
        chunks,
        embedder.embed([chunk.text for chunk in chunks]),
        embedder.dimensions,
        client=client,
        collection="dense_test",
    )


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


def test_provider_filter_excludes_other_providers() -> None:
    client = QdrantClient(":memory:")
    embedder = _FixedEmbedder()
    _index(client, embedder)
    retriever = DenseRetriever(
        embedder=embedder,
        client=client,
        collection="dense_test",
        top_k=5,
    )
    results = retriever.search(
        "How much does RingCentral cost?",
        filters=RetrievalFilters(provider="Nextiva"),
    )
    assert results
    assert {chunk.provider for chunk in results} == {"Nextiva"}


def test_section_and_provider_filter() -> None:
    client = QdrantClient(":memory:")
    embedder = _FixedEmbedder()
    _index(client, embedder)
    retriever = DenseRetriever(
        embedder=embedder,
        client=client,
        collection="dense_test",
        top_k=5,
    )
    results = retriever.search(
        "How much does Nextiva cost?",
        filters=RetrievalFilters(provider="Nextiva", section="Pricing"),
    )
    assert [chunk.section for chunk in results] == ["Pricing"]
    assert results[0].provider == "Nextiva"
    assert results[0].document_id == "8019"


def test_infer_filters_from_ringcentral_pricing_query() -> None:
    client = QdrantClient(":memory:")
    embedder = _FixedEmbedder()
    _index(client, embedder)
    retriever = DenseRetriever(
        embedder=embedder,
        client=client,
        collection="dense_test",
        top_k=5,
    )
    results = retriever.search("How much does RingCentral cost?", infer=True)
    assert results
    assert {chunk.provider for chunk in results} == {"RingCentral"}
    assert all("pricing" in chunk.section.lower() for chunk in results)
