from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.ingestion.push import push_collections


def test_push_copies_points_to_destination(monkeypatch) -> None:
    source = QdrantClient(":memory:")
    dest = QdrantClient(":memory:")
    source.create_collection(
        collection_name="getvoip_chunks_structure_aware",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    source.upsert(
        collection_name="getvoip_chunks_structure_aware",
        points=[PointStruct(id=1, vector=[0.1, 0.2], payload={"document_id": "8019"})],
    )

    def fake_client(url="http://127.0.0.1:6333", api_key=None):
        return source if "6333" in url else dest

    monkeypatch.setattr("app.ingestion.push.get_qdrant_client", fake_client)
    monkeypatch.setattr(
        "app.ingestion.push.QDRANT_COLLECTIONS",
        {"structure_aware": "getvoip_chunks_structure_aware"},
    )
    monkeypatch.setattr("app.ingestion.push.QDRANT_COLLECTION_RAPTOR", "getvoip_chunks_raptor")

    copied = push_collections("http://127.0.0.1:6333", "https://qdrant.example")
    assert copied["getvoip_chunks_structure_aware"] == 1
    points, _ = dest.scroll("getvoip_chunks_structure_aware", limit=5)
    assert points[0].payload["document_id"] == "8019"


def test_push_rejects_same_url() -> None:
    try:
        push_collections("http://127.0.0.1:6333", "http://127.0.0.1:6333")
    except ValueError as exc:
        assert "same" in str(exc)
    else:
        raise AssertionError("expected ValueError")
