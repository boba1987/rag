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
        "parent_id": None,
    }


def test_payload_roundtrip() -> None:
    assert chunk_from_payload(chunk_payload(_CHUNK)) == _CHUNK


def test_point_id_is_stable() -> None:
    assert chunk_point_id(_CHUNK.id) == chunk_point_id(_CHUNK.id)
    assert chunk_point_id(_CHUNK.id) != chunk_point_id("provider_8015_pricing_01")


class _RecordingClient:
    def __init__(self) -> None:
        self.sizes: list[int] = []

    def collection_exists(self, name: str) -> bool:
        return True

    def upsert(self, collection_name: str, points, timeout=None) -> None:
        self.sizes.append(len(points))


def test_upsert_sends_points_in_batches(monkeypatch) -> None:
    monkeypatch.setattr("app.ingestion.indexer.QDRANT_UPSERT_BATCH", 2)
    chunks = [_CHUNK.model_copy(update={"id": f"{_CHUNK.id}_{index}"}) for index in range(5)]
    vectors = [[0.1, 0.2, 0.3, 0.4] for _ in chunks]
    client = _RecordingClient()
    assert upsert_chunks(chunks, vectors, 4, client=client, collection="batched") == 5
    assert client.sizes == [2, 2, 1]


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
