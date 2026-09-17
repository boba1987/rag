from app.config import CHUNKER_NAMES
from app.ingestion.ingest import main


class _FakeEmbedder:
    dimensions = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, int] = {}

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, collection_name: str, vectors_config) -> None:
        self.collections[collection_name] = 0

    def upsert(self, collection_name: str, points) -> None:
        self.collections[collection_name] = self.collections.get(collection_name, 0) + len(points)


def test_ingest_all_writes_each_chunker(tmp_path, monkeypatch) -> None:
    source = tmp_path / "providers.json"
    source.write_text(
        '[{"id": 1, "name": "Acme", "slug": "acme", "content": "<h2>Pricing</h2><p>Ten dollars.</p>"}]\n',
        encoding="utf-8",
    )
    client = _FakeClient()
    monkeypatch.setattr(
        "app.ingestion.ingest.chunked_dir",
        lambda name=None: tmp_path / f"chunked_{name or 'active'}",
    )
    monkeypatch.setattr(
        "app.ingestion.ingest.qdrant_collection",
        lambda name=None: f"col_{name or 'active'}",
    )
    monkeypatch.setattr("app.ingestion.ingest.INDEXED_DIR", tmp_path / "indexed")
    monkeypatch.setattr("app.ingestion.ingest.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr("app.ingestion.ingest.get_qdrant_client", lambda: client)
    monkeypatch.setattr("app.ingestion.normalizer.NORMALIZED_DIR", tmp_path / "normalized")

    assert main([str(source), "provider", "--chunker", "all"]) == 0
    for name in CHUNKER_NAMES:
        assert (tmp_path / f"chunked_{name}" / "provider-1-acme.json").exists()
        assert (tmp_path / "indexed" / f"last-run-{name}.json").exists()
        assert client.collections[f"col_{name}"] > 0
    assert (tmp_path / "chunked_parent_child" / "parents" / "provider-1-acme.json").exists()
