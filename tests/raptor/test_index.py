import json

from qdrant_client import QdrantClient

from app.config import QDRANT_COLLECTION_RAPTOR
from app.ingestion.embedder import Embedder
from app.models.schemas import Chunk
from app.raptor.index import (
    index_raptor_document,
    main,
    node_from_payload,
    raptor_payload,
    upsert_raptor_nodes,
)
from app.raptor.summarization import ConcatSummarizer
from app.raptor.tree import RaptorNode, leaf_from_chunk


class _FakeEmbedder(Embedder):
    dimensions = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for index, _text in enumerate(texts):
            vectors.append([1.0 - (index % 2) * 0.9, 0.1 + (index % 2) * 0.8])
        return vectors


def _chunk(**overrides) -> Chunk:
    data = dict(
        id="review_100_pricing_01",
        document_id="100",
        content_type="review",
        provider="RingCentral",
        title="RingCentral Review",
        section="Pricing",
        heading_path=["RingCentral Review", "Pricing"],
        text="Core starts at $30 per user.",
        source_url="https://example.test/ringcentral",
    )
    data.update(overrides)
    return Chunk(**data)


def _chunks() -> list[Chunk]:
    return [
        _chunk(id="pricing", section="Pricing", text="Core starts at $30."),
        _chunk(id="features", section="Features", text="Includes video meetings."),
        _chunk(id="support", section="Support", text="Phone support is 24/7."),
        _chunk(id="reviews", section="Reviews", text="Users praise call quality."),
    ]


def test_raptor_collection_is_separate_from_chunkers() -> None:
    assert QDRANT_COLLECTION_RAPTOR == "getvoip_chunks_raptor"


def test_raptor_payload_includes_tree_fields() -> None:
    node = RaptorNode(
        id="raptor_100_l1_00",
        level=1,
        node_type="summary",
        text="Pricing and features together.",
        children=["pricing", "features"],
        document_id="100",
        content_type="review",
        provider="RingCentral",
        title="RingCentral Review",
        section="Summary",
        heading_path=["RingCentral Review", "Summary"],
        source_url="https://example.test/ringcentral",
    )
    payload = raptor_payload(node)
    assert payload["chunk_id"] == "raptor_100_l1_00"
    assert payload["node_type"] == "summary"
    assert payload["level"] == 1
    assert payload["children"] == ["pricing", "features"]
    assert node_from_payload(payload) == node


def test_leaf_payload_roundtrip() -> None:
    leaf = leaf_from_chunk(_chunk())
    assert node_from_payload(raptor_payload(leaf)).id == leaf.id
    assert node_from_payload(raptor_payload(leaf)).node_type == "leaf"


def test_upsert_raptor_nodes_in_memory() -> None:
    client = QdrantClient(":memory:")
    parent = RaptorNode(
        id="raptor_100_l1_00",
        level=1,
        node_type="summary",
        text="Core is $30 and includes video.",
        children=["pricing"],
        document_id="100",
        content_type="review",
        title="RingCentral Review",
        section="Summary",
        heading_path=["RingCentral Review", "Summary"],
    )
    leaf = leaf_from_chunk(_chunk(id="pricing"))
    leaf.parent_id = parent.id
    upserted = upsert_raptor_nodes(
        [parent, leaf],
        [[1.0, 0.0], [0.0, 1.0]],
        2,
        client=client,
        collection=QDRANT_COLLECTION_RAPTOR,
    )
    assert upserted == 2
    points, _ = client.scroll(QDRANT_COLLECTION_RAPTOR, limit=5, with_vectors=False)
    payloads = {point.payload["chunk_id"]: point.payload for point in points}
    assert set(payloads) == {"raptor_100_l1_00", "pricing"}
    assert payloads["raptor_100_l1_00"]["node_type"] == "summary"
    assert payloads["pricing"]["node_type"] == "leaf"
    assert payloads["pricing"]["parent_id"] == "raptor_100_l1_00"


def test_index_raptor_document_writes_tree_and_upserts(tmp_path) -> None:
    client = QdrantClient(":memory:")
    tree_path = tmp_path / "review-100.json"
    tree, upserted = index_raptor_document(
        _chunks(),
        embedder=_FakeEmbedder(),
        summarizer=ConcatSummarizer(),
        client=client,
        collection=QDRANT_COLLECTION_RAPTOR,
        tree_path=tree_path,
    )
    assert upserted == len(tree.nodes)
    assert upserted > 4
    assert any(node.node_type == "summary" for node in tree.nodes.values())
    assert tree_path.exists()
    stored = json.loads(tree_path.read_text(encoding="utf-8"))
    assert {row["id"] for row in stored} == set(tree.nodes)
    points, _ = client.scroll(QDRANT_COLLECTION_RAPTOR, limit=20, with_vectors=False)
    assert len(points) == upserted
    assert {point.payload["node_type"] for point in points} == {"leaf", "summary"}


def test_raptor_index_cli_all(tmp_path, monkeypatch) -> None:
    source = tmp_path / "chunked"
    source.mkdir()
    (source / "review-100.json").write_text(
        json.dumps([chunk.model_dump() for chunk in _chunks()]) + "\n",
        encoding="utf-8",
    )
    client = QdrantClient(":memory:")
    monkeypatch.setattr("app.raptor.index.chunked_dir", lambda name=None: source)
    monkeypatch.setattr("app.raptor.index.RAPTOR_DIR", tmp_path / "raptor")
    monkeypatch.setattr("app.raptor.index.INDEXED_DIR", tmp_path / "indexed")
    monkeypatch.setattr("app.raptor.index.QDRANT_COLLECTION_RAPTOR", "getvoip_chunks_raptor")
    monkeypatch.setattr("app.raptor.index.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr("app.raptor.index.get_summarizer", ConcatSummarizer)
    monkeypatch.setattr("app.raptor.index.get_qdrant_client", lambda: client)

    assert main(["--all"]) == 0
    assert (tmp_path / "raptor" / "review-100.json").exists()
    report = json.loads((tmp_path / "indexed" / "last-run-raptor.json").read_text(encoding="utf-8"))
    assert report["collection"] == "getvoip_chunks_raptor"
    assert report["strategy"] == "raptor"
    assert report["node_count"] > 4
    assert report["documents"][0]["leaves"] == 4
    assert report["documents"][0]["summaries"] >= 1
    points, _ = client.scroll("getvoip_chunks_raptor", limit=20, with_vectors=False)
    assert len(points) == report["node_count"]


def test_raptor_index_cli_ensure_collection(monkeypatch) -> None:
    client = QdrantClient(":memory:")
    monkeypatch.setattr("app.raptor.index.get_qdrant_client", lambda: client)
    monkeypatch.setattr("app.raptor.index.QDRANT_COLLECTION_RAPTOR", "getvoip_chunks_raptor")
    monkeypatch.setattr("app.raptor.index.OPENAI_EMBED_DIMENSIONS", 2)

    assert main(["--ensure-collection"]) == 0
    assert client.collection_exists("getvoip_chunks_raptor")
