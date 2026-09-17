from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from app.models.schemas import Chunk, ContentType
from app.raptor.clustering import cluster_embeddings


class RaptorNode(BaseModel):
    """A leaf chunk or a later summary node in the RAPTOR tree."""

    id: str
    level: int = 0
    node_type: Literal["leaf", "summary"] = "leaf"
    text: str = ""
    children: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    document_id: str | None = None
    content_type: ContentType | None = None
    provider: str | None = None
    title: str = ""
    section: str = ""
    heading_path: list[str] = Field(default_factory=list)
    source_url: str | None = None


class RaptorTree(BaseModel):
    nodes: dict[str, RaptorNode] = Field(default_factory=dict)

    def add(self, node: RaptorNode) -> None:
        self.nodes[node.id] = node

    def leaves(self) -> list[RaptorNode]:
        return [node for node in self.nodes.values() if node.node_type == "leaf"]

    def roots(self) -> list[RaptorNode]:
        return [node for node in self.nodes.values() if node.parent_id is None]

    def children_of(self, node_id: str) -> list[RaptorNode]:
        node = self.nodes[node_id]
        return [self.nodes[child_id] for child_id in node.children if child_id in self.nodes]


def leaf_from_chunk(chunk: Chunk) -> RaptorNode:
    """Promote a structure-aware chunk to a level-0 RAPTOR leaf."""
    return RaptorNode(
        id=chunk.id,
        level=0,
        node_type="leaf",
        text=chunk.text,
        document_id=chunk.document_id,
        content_type=chunk.content_type,
        provider=chunk.provider,
        title=chunk.title,
        section=chunk.section,
        heading_path=list(chunk.heading_path),
        source_url=chunk.source_url,
    )


def _parent_id(document_id: str | None, level: int, index: int) -> str:
    doc = document_id or "mixed"
    return f"raptor_{doc}_l{level}_{index:02d}"


def _cluster_indices(
    nodes: list[RaptorNode],
    vectors: list[list[float]],
    *,
    k: int | None,
    by_document: bool,
) -> list[list[int]]:
    if not by_document:
        return cluster_embeddings(vectors, k)

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        grouped[node.document_id or "mixed"].append(index)

    clusters: list[list[int]] = []
    for document_id in sorted(grouped):
        member_indices = grouped[document_id]
        local_vectors = [vectors[index] for index in member_indices]
        for local_group in cluster_embeddings(local_vectors, k):
            clusters.append([member_indices[local] for local in local_group])
    return clusters


def build_tree_level(
    nodes: list[RaptorNode],
    vectors: list[list[float]],
    *,
    k: int | None = None,
    by_document: bool = True,
) -> list[RaptorNode]:
    """Cluster nodes and attach empty-text summary parents. Returns parents + children."""
    if len(nodes) != len(vectors):
        raise ValueError("Each RAPTOR node needs a matching embedding")
    if not nodes:
        return []

    level = max(node.level for node in nodes) + 1
    updated = [node.model_copy(deep=True) for node in nodes]
    parents: list[RaptorNode] = []

    for index, group in enumerate(_cluster_indices(updated, vectors, k=k, by_document=by_document)):
        children = [updated[member] for member in group]
        first = children[0]
        same_document = all(child.document_id == first.document_id for child in children)
        same_provider = same_document and all(child.provider == first.provider for child in children)
        document_id = first.document_id if same_document else None
        parent = RaptorNode(
            id=_parent_id(document_id, level, index),
            level=level,
            node_type="summary",
            text="",
            children=[child.id for child in children],
            document_id=document_id,
            content_type=first.content_type if same_document else None,
            provider=first.provider if same_provider else None,
            title=first.title if same_document else "",
            section="Summary",
            heading_path=[first.title, "Summary"] if first.title else ["Summary"],
            source_url=first.source_url if same_document else None,
        )
        for child in children:
            child.parent_id = parent.id
        parents.append(parent)

    return parents + updated
