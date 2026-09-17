import pytest
from qdrant_client import QdrantClient

from app.config import chunked_dir, qdrant_collection
from app.ingestion.indexer import ensure_chunker_collections


def test_each_chunker_has_its_own_collection_and_directory() -> None:
    collections = {name: qdrant_collection(name) for name in ("structure_aware", "fixed_size", "parent_child")}
    directories = {name: chunked_dir(name) for name in collections}
    assert collections["structure_aware"] == "getvoip_chunks_structure_aware"
    assert collections["fixed_size"] == "getvoip_chunks_fixed"
    assert collections["parent_child"] == "getvoip_chunks_parent_child"
    assert len(set(collections.values())) == 3
    assert len(set(directories.values())) == 3
    assert directories["structure_aware"].name == "chunked"


def test_unknown_chunker_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown chunker"):
        qdrant_collection("raptor")


def test_ensure_chunker_collections_creates_all_three() -> None:
    client = QdrantClient(":memory:")
    created = ensure_chunker_collections(client, dimensions=4)
    assert set(created) == {"structure_aware", "fixed_size", "parent_child"}
    for collection in created.values():
        assert client.collection_exists(collection)
