from qdrant_client import QdrantClient

from app.config import QDRANT_COLLECTION_RAPTOR
from app.ingestion.embedder import Embedder
from app.models.schemas import RetrievalFilters, RetrievedChunk
from app.raptor.index import upsert_raptor_nodes
from app.raptor.retrieval import (
    RaptorHit,
    RaptorRetriever,
    collapse_raptor_hits,
    retrieved_from_raptor_payload,
)
from app.raptor.tree import RaptorNode


class _FixedEmbedder(Embedder):
    dimensions = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        mapping = {
            "What are the main strengths of RingCentral?": [1.0, 0.0],
            "How much does RingCentral cost?": [0.0, 1.0],
        }
        return [mapping.get(text, [0.5, 0.5]) for text in texts]


def _chunk(**overrides) -> RetrievedChunk:
    data = dict(
        id="pricing",
        document_id="100",
        content_type="review",
        provider="RingCentral",
        title="RingCentral Review",
        section="Pricing",
        heading_path=["RingCentral Review", "Pricing"],
        text="Core starts at $30.",
        score=0.8,
    )
    data.update(overrides)
    return RetrievedChunk(**data)


def test_retrieved_from_raptor_payload_fills_mixed_defaults() -> None:
    chunk = retrieved_from_raptor_payload(
        {
            "chunk_id": "raptor_mixed_l1_00",
            "text": "Two providers compared.",
            "node_type": "summary",
        },
        0.42,
    )
    assert chunk.document_id == "mixed"
    assert chunk.content_type == "article"
    assert chunk.section == "Summary"
    assert chunk.heading_path == ["Summary"]
    assert chunk.score == 0.42


def test_collapse_drops_children_when_summary_is_present() -> None:
    parent = RaptorHit(
        chunk=_chunk(id="summary", section="Summary", text="Overview of pricing.", score=0.9),
        node_type="summary",
        children=("pricing",),
        level=1,
    )
    child = RaptorHit(
        chunk=_chunk(score=0.85),
        node_type="leaf",
        children=(),
        level=0,
    )
    collapsed = collapse_raptor_hits([parent, child])
    assert [chunk.id for chunk in collapsed] == ["summary"]


def test_collapse_keeps_leaves_when_no_parent_hit() -> None:
    child = RaptorHit(chunk=_chunk(), node_type="leaf", children=(), level=0)
    assert [chunk.id for chunk in collapse_raptor_hits([child])] == ["pricing"]


def test_raptor_search_collapses_overlapping_tree_hits() -> None:
    client = QdrantClient(":memory:")
    parent = RaptorNode(
        id="raptor_100_l1_00",
        level=1,
        node_type="summary",
        text="RingCentral is strong on pricing and reliability.",
        children=["pricing"],
        document_id="100",
        content_type="review",
        provider="RingCentral",
        title="RingCentral Review",
        section="Summary",
        heading_path=["RingCentral Review", "Summary"],
    )
    leaf = RaptorNode(
        id="pricing",
        text="Core starts at $30 per user.",
        parent_id=parent.id,
        document_id="100",
        content_type="review",
        provider="RingCentral",
        title="RingCentral Review",
        section="Pricing",
        heading_path=["RingCentral Review", "Pricing"],
    )
    upsert_raptor_nodes(
        [parent, leaf],
        [[1.0, 0.0], [0.2, 0.8]],
        2,
        client=client,
        collection=QDRANT_COLLECTION_RAPTOR,
    )
    retriever = RaptorRetriever(
        embedder=_FixedEmbedder(),
        client=client,
        collection=QDRANT_COLLECTION_RAPTOR,
        top_k=2,
    )
    results = retriever.search("What are the main strengths of RingCentral?")
    assert [chunk.id for chunk in results] == ["raptor_100_l1_00"]
    assert "strong on pricing" in results[0].text


def test_raptor_search_respects_provider_filter() -> None:
    client = QdrantClient(":memory:")
    ringcentral = RaptorNode(
        id="rc",
        text="RingCentral Core is $30.",
        document_id="100",
        content_type="review",
        provider="RingCentral",
        title="RingCentral Review",
        section="Pricing",
        heading_path=["RingCentral Review", "Pricing"],
    )
    nextiva = RaptorNode(
        id="nx",
        text="Nextiva Core is $15.",
        document_id="200",
        content_type="review",
        provider="Nextiva",
        title="Nextiva Review",
        section="Pricing",
        heading_path=["Nextiva Review", "Pricing"],
    )
    upsert_raptor_nodes(
        [ringcentral, nextiva],
        [[1.0, 0.0], [0.9, 0.1]],
        2,
        client=client,
        collection="raptor_filter",
    )
    retriever = RaptorRetriever(
        embedder=_FixedEmbedder(),
        client=client,
        collection="raptor_filter",
        top_k=5,
    )
    results = retriever.search(
        "What are the main strengths of RingCentral?",
        filters=RetrievalFilters(provider="Nextiva"),
    )
    assert results
    assert {chunk.provider for chunk in results} == {"Nextiva"}
