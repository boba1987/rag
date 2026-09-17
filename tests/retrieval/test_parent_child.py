from app.models.schemas import RetrievedChunk
from app.retrieval.parent_child import FileParentStore, ParentExpandingRetriever, expand_to_parents


def _child(**overrides) -> RetrievedChunk:
    data = dict(
        id="provider_8015_pricing_child_01",
        document_id="8015",
        content_type="provider",
        provider="RingCentral",
        title="RingCentral",
        section="Plans and Pricing",
        heading_path=["RingCentral", "Plans and Pricing"],
        text="Core starts at a short child snippet.",
        source_url="https://example.test/ringcentral",
        parent_id="provider_8015_pricing_parent",
        score=0.9,
    )
    data.update(overrides)
    return RetrievedChunk(**data)


def _parent_store(tmp_path) -> FileParentStore:
    directory = tmp_path / "parents"
    directory.mkdir()
    (directory / "provider-8015.json").write_text(
        """
        [
          {
            "id": "provider_8015_pricing_parent",
            "document_id": "8015",
            "content_type": "provider",
            "provider": "RingCentral",
            "title": "RingCentral",
            "section": "Plans and Pricing",
            "heading_path": ["RingCentral", "Plans and Pricing"],
            "text": "Full parent pricing section with Core, Engage, and annual billing.",
            "source_url": "https://example.test/ringcentral",
            "parent_id": null
          }
        ]
        """
    )
    return FileParentStore(directory)


def test_expands_child_hit_to_parent_text(tmp_path) -> None:
    store = _parent_store(tmp_path)
    expanded = expand_to_parents([_child()], store)
    assert len(expanded) == 1
    assert expanded[0].id == "provider_8015_pricing_parent"
    assert expanded[0].parent_id is None
    assert "annual billing" in expanded[0].text
    assert expanded[0].score == 0.9


def test_sibling_children_collapse_to_one_parent(tmp_path) -> None:
    store = _parent_store(tmp_path)
    first = _child(id="provider_8015_pricing_child_01", score=0.95)
    second = _child(id="provider_8015_pricing_child_02", score=0.8)
    expanded = expand_to_parents([first, second], store)
    assert [chunk.id for chunk in expanded] == ["provider_8015_pricing_parent"]
    assert expanded[0].score == 0.95


def test_missing_parent_keeps_child(tmp_path) -> None:
    store = FileParentStore(tmp_path / "empty")
    child = _child()
    assert expand_to_parents([child], store) == [child]


def test_chunk_without_parent_id_is_unchanged(tmp_path) -> None:
    store = _parent_store(tmp_path)
    lone = _child(parent_id=None, id="structure_aware_01")
    assert expand_to_parents([lone], store) == [lone]


def test_parent_expanding_retriever_searches_then_expands(tmp_path) -> None:
    class _Inner:
        def search(self, query: str, top_k=None, filters=None, infer: bool = False):
            assert query == "RingCentral pricing"
            return [_child()]

    retriever = ParentExpandingRetriever(_Inner(), store=_parent_store(tmp_path))
    hits = retriever.search("RingCentral pricing")
    assert hits[0].id == "provider_8015_pricing_parent"
