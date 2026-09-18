from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from app.config import (
    EMBEDDER_PROVIDER,
    INDEXED_DIR,
    OPENAI_EMBED_DIMENSIONS,
    OPENAI_EMBED_MODEL,
    QDRANT_COLLECTION_RAPTOR,
    RAPTOR_DIR,
    RAPTOR_SUMMARIZER,
    chunked_dir,
)
from app.ingestion.embed_text import build_embedding_text
from app.ingestion.embedder import Embedder, get_embedder
from app.ingestion.index import _embed_batches, _load_chunk_files
from app.ingestion.indexer import (
    chunk_point_id,
    ensure_collection,
    get_qdrant_client,
    upsert_points,
    write_index_report,
)
from app.models.schemas import Chunk
from app.raptor.summarization import (
    ClusterSummarizer,
    build_raptor_tree,
    get_summarizer,
    raptor_embed_text,
)
from app.raptor.tree import RaptorNode, RaptorTree, leaf_from_chunk


def raptor_payload(node: RaptorNode) -> dict:
    return {
        "chunk_id": node.id,
        "document_id": node.document_id,
        "content_type": node.content_type,
        "provider": node.provider,
        "title": node.title,
        "section": node.section or "Summary",
        "heading_path": list(node.heading_path) or [node.section or "Summary"],
        "text": node.text,
        "source_url": node.source_url,
        "parent_id": node.parent_id,
        "node_type": node.node_type,
        "level": node.level,
        "children": list(node.children),
    }


def node_from_payload(payload: dict) -> RaptorNode:
    return RaptorNode(
        id=payload["chunk_id"],
        level=int(payload.get("level") or 0),
        node_type=payload.get("node_type") or "leaf",
        text=payload.get("text") or "",
        children=list(payload.get("children") or []),
        parent_id=payload.get("parent_id"),
        document_id=payload.get("document_id"),
        content_type=payload.get("content_type"),
        provider=payload.get("provider"),
        title=payload.get("title") or "",
        section=payload.get("section") or "",
        heading_path=list(payload.get("heading_path") or [payload.get("section") or "Summary"]),
        source_url=payload.get("source_url"),
    )


def write_raptor_tree(tree: RaptorTree, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [node.model_dump() for node in tree.nodes.values()],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def upsert_raptor_nodes(
    nodes: list[RaptorNode],
    vectors: list[list[float]],
    dimensions: int,
    client: QdrantClient | None = None,
    collection: str = QDRANT_COLLECTION_RAPTOR,
) -> int:
    if len(nodes) != len(vectors):
        raise ValueError("nodes and vectors must be the same length")
    qdrant = client or get_qdrant_client()
    ensure_collection(qdrant, dimensions, collection)
    points = [
        PointStruct(
            id=chunk_point_id(node.id),
            vector=vector,
            payload=raptor_payload(node),
        )
        for node, vector in zip(nodes, vectors, strict=True)
    ]
    upsert_points(qdrant, collection, points)
    return len(points)


def _vectors_for_tree(
    tree: RaptorTree,
    leaf_vectors: dict[str, list[float]],
    embedder: Embedder,
) -> list[tuple[RaptorNode, list[float]]]:
    nodes = list(tree.nodes.values())
    known = dict(leaf_vectors)
    missing = [node for node in nodes if node.id not in known]
    if missing:
        extra = _embed_batches(embedder, [raptor_embed_text(node) for node in missing])
        known.update({node.id: vector for node, vector in zip(missing, extra, strict=True)})
    return [(node, known[node.id]) for node in nodes]


def index_raptor_document(
    chunks: list[Chunk],
    *,
    embedder: Embedder,
    summarizer: ClusterSummarizer,
    client: QdrantClient,
    collection: str = QDRANT_COLLECTION_RAPTOR,
    tree_path: Path | None = None,
) -> tuple[RaptorTree, int]:
    """Build a RAPTOR tree from structure-aware chunks and upsert every node."""
    leaves = [leaf_from_chunk(chunk) for chunk in chunks]
    leaf_vectors = _embed_batches(embedder, [build_embedding_text(chunk) for chunk in chunks])
    tree = build_raptor_tree(
        leaves,
        leaf_vectors,
        summarizer=summarizer,
        embedder=embedder,
    )
    paired = _vectors_for_tree(
        tree,
        {leaf.id: vector for leaf, vector in zip(leaves, leaf_vectors, strict=True)},
        embedder,
    )
    nodes = [node for node, _vector in paired]
    vectors = [vector for _node, vector in paired]
    upserted = upsert_raptor_nodes(
        nodes,
        vectors,
        embedder.dimensions,
        client=client,
        collection=collection,
    )
    if tree_path is not None:
        write_raptor_tree(tree, tree_path)
    return tree, upserted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a RAPTOR tree from structure-aware chunks and index it in Qdrant."
    )
    parser.add_argument("--id", dest="post_id", help="WordPress post id")
    parser.add_argument(
        "--type",
        dest="content_type",
        choices=("article", "review", "provider"),
        help="Document content type",
    )
    parser.add_argument("--all", action="store_true", help="Index every structure-aware chunk file")
    parser.add_argument(
        "--ensure-collection",
        action="store_true",
        help="Create the RAPTOR Qdrant collection and exit",
    )
    args = parser.parse_args(argv)

    client = get_qdrant_client()
    if args.ensure_collection:
        ensure_collection(client, OPENAI_EMBED_DIMENSIONS, QDRANT_COLLECTION_RAPTOR)
        print(f"raptor → {QDRANT_COLLECTION_RAPTOR}")
        return 0

    if not args.all and not args.post_id:
        parser.error("Provide --id, --all, or --ensure-collection")

    files = _load_chunk_files(
        post_id=args.post_id,
        content_type=args.content_type,
        directory=chunked_dir("structure_aware"),
    )
    if not files:
        print("No structure-aware chunk files matched. Run rag-chunk first.", file=sys.stderr)
        return 1

    embedder = get_embedder()
    summarizer = get_summarizer()
    documents = []
    total = 0

    for path, chunks in files:
        tree_path = RAPTOR_DIR / path.name
        tree, upserted = index_raptor_document(
            chunks,
            embedder=embedder,
            summarizer=summarizer,
            client=client,
            collection=QDRANT_COLLECTION_RAPTOR,
            tree_path=tree_path,
        )
        total += upserted
        leaves = sum(1 for node in tree.nodes.values() if node.node_type == "leaf")
        summaries = upserted - leaves
        documents.append(
            {
                "id": chunks[0].document_id,
                "content_type": chunks[0].content_type,
                "source": path.name,
                "nodes": upserted,
                "leaves": leaves,
                "summaries": summaries,
                "status": "ok",
            }
        )
        print(f"{path.name}  ({leaves} leaves + {summaries} summaries → {QDRANT_COLLECTION_RAPTOR})")

    report_path = write_index_report(
        {
            "collection": QDRANT_COLLECTION_RAPTOR,
            "chunker": "structure_aware",
            "strategy": "raptor",
            "embedder": EMBEDDER_PROVIDER,
            "summarizer": RAPTOR_SUMMARIZER,
            "model": OPENAI_EMBED_MODEL if EMBEDDER_PROVIDER == "openai" else EMBEDDER_PROVIDER,
            "node_count": total,
            "documents": documents,
        },
        path=INDEXED_DIR / "last-run-raptor.json",
    )
    print(f"report {report_path}  ({total} nodes total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
