from qdrant_client import QdrantClient

from app.ingestion.indexer import chunk_from_payload, chunk_payload, chunk_point_id, upsert_chunks
from app.models.schemas import Chunk

_CHUNK = Chunk(
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


class _FakeEmbedder:
    dimensions = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def test_payload_includes_planned_metadata() -> None:
    payload = chunk_payload(_CHUNK)
    assert payload == {
        "chunk_id": "provider_8019_pricing_01",
        "document_id": "8019",
        "content_type": "provider",
        "provider": "Nextiva",
        "title": "Nextiva",
        "section": "Pricing",
        "heading_path": ["Nextiva", "Pricing"],
        "text": "Core starts at $15 per user.",
        "source_url": "http://localhost/review/nextiva-2/",
        "updated_at": "2026-09-02 06:31:16",
    }


def test_payload_roundtrip() -> None:
    assert chunk_from_payload(chunk_payload(_CHUNK)) == _CHUNK


def test_point_id_is_stable() -> None:
    assert chunk_point_id(_CHUNK.id) == chunk_point_id(_CHUNK.id)
    assert chunk_point_id(_CHUNK.id) != chunk_point_id("provider_8015_pricing_01")


def test_upsert_roundtrip_in_memory() -> None:
    client = QdrantClient(":memory:")
    embedder = _FakeEmbedder()
    upserted = upsert_chunks(
        [_CHUNK],
        embedder.embed([_CHUNK.text]),
        embedder.dimensions,
        client=client,
        collection="inspect_chunks",
    )
    assert upserted == 1

    points, _ = client.scroll("inspect_chunks", limit=5, with_vectors=False)
    assert len(points) == 1
    payload = points[0].payload
    assert payload["section"] == "Pricing"
    assert payload["updated_at"] == "2026-09-02 06:31:16"
    assert "Core starts at $15" in payload["text"]
