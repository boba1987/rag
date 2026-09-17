import pytest

from app.models.schemas import Chunk
from app.raptor.tree import RaptorNode, RaptorTree, build_tree_level, leaf_from_chunk


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


def test_leaf_from_chunk_copies_retrieval_metadata() -> None:
    leaf = leaf_from_chunk(_chunk())
    assert leaf.node_type == "leaf"
    assert leaf.level == 0
    assert leaf.text == "Core starts at $30 per user."
    assert leaf.document_id == "100"
    assert leaf.provider == "RingCentral"
    assert leaf.heading_path == ["RingCentral Review", "Pricing"]


def test_build_tree_level_requires_matching_vectors() -> None:
    with pytest.raises(ValueError, match="matching embedding"):
        build_tree_level([leaf_from_chunk(_chunk())], [])


def test_build_tree_level_attaches_empty_summary_parents() -> None:
    leaves = [
        leaf_from_chunk(_chunk(id="pricing", section="Pricing", text="price")),
        leaf_from_chunk(_chunk(id="features", section="Features", text="features")),
        leaf_from_chunk(_chunk(id="support", section="Support", text="support")),
        leaf_from_chunk(_chunk(id="reviews", section="Reviews", text="reviews")),
    ]
    vectors = [[1.0, 0.0], [0.95, 0.1], [0.0, 1.0], [0.1, 0.95]]
    nodes = build_tree_level(leaves, vectors, k=2)
    parents = [node for node in nodes if node.node_type == "summary"]
    children = [node for node in nodes if node.node_type == "leaf"]

    assert len(parents) == 2
    assert all(parent.text == "" for parent in parents)
    assert all(parent.level == 1 for parent in parents)
    assert all(parent.section == "Summary" for parent in parents)
    assert {child.parent_id for child in children} == {parent.id for parent in parents}
    assert sorted(child_id for parent in parents for child_id in parent.children) == [
        "features",
        "pricing",
        "reviews",
        "support",
    ]


def test_build_tree_level_keeps_documents_separate() -> None:
    leaves = [
        leaf_from_chunk(_chunk(id="rc_price", document_id="100", title="RingCentral Review")),
        leaf_from_chunk(_chunk(id="rc_feat", document_id="100", title="RingCentral Review")),
        leaf_from_chunk(
            _chunk(
                id="nx_price",
                document_id="200",
                provider="Nextiva",
                title="Nextiva Review",
            )
        ),
        leaf_from_chunk(
            _chunk(
                id="nx_feat",
                document_id="200",
                provider="Nextiva",
                title="Nextiva Review",
            )
        ),
    ]
    similar = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    nodes = build_tree_level(leaves, similar, k=1, by_document=True)
    parents = [node for node in nodes if node.node_type == "summary"]

    assert {parent.document_id for parent in parents} == {"100", "200"}
    ringcentral = next(parent for parent in parents if parent.document_id == "100")
    nextiva = next(parent for parent in parents if parent.document_id == "200")
    assert set(ringcentral.children) == {"rc_price", "rc_feat"}
    assert set(nextiva.children) == {"nx_price", "nx_feat"}
    assert ringcentral.provider == "RingCentral"
    assert nextiva.provider == "Nextiva"


def test_raptor_tree_roots_and_children() -> None:
    parent = RaptorNode(id="root", level=1, node_type="summary", children=["leaf"])
    leaf = RaptorNode(id="leaf", parent_id="root", text="body")
    tree = RaptorTree(nodes={parent.id: parent, leaf.id: leaf})

    assert [node.id for node in tree.roots()] == ["root"]
    assert [node.id for node in tree.leaves()] == ["leaf"]
    assert [node.id for node in tree.children_of("root")] == ["leaf"]
