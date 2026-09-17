from __future__ import annotations

from abc import ABC, abstractmethod

from openai import OpenAI

from app.config import (
    OPENAI_API_KEY,
    RAPTOR_MAX_LEVELS,
    RAPTOR_SUMMARIZER,
    RAPTOR_SUMMARIZER_MODEL,
)
from app.ingestion.embedder import Embedder
from app.raptor.clustering import choose_cluster_count
from app.raptor.tree import RaptorNode, RaptorTree, build_tree_level

_SYSTEM = (
    "Summarize related GetVoIP knowledge-base passages into one paragraph. "
    "Keep provider names, pricing, features, and other facts from the passages. "
    "Do not invent details that are not present."
)


class ClusterSummarizer(ABC):
    """Swappable cluster summarizer. OpenAI now; Bedrock later."""

    @abstractmethod
    def summarize(self, parent: RaptorNode, children: list[RaptorNode]) -> str:
        raise NotImplementedError


class ConcatSummarizer(ClusterSummarizer):
    """Offline fallback: join child headings and text. No LLM."""

    def summarize(self, parent: RaptorNode, children: list[RaptorNode]) -> str:
        parts = []
        for child in children:
            heading = child.section or child.title or parent.title
            body = child.text.strip()
            parts.append(f"{heading}: {body}".strip(": ").strip())
        return " ".join(part for part in parts if part)


class OpenAIClusterSummarizer(ClusterSummarizer):
    def __init__(
        self,
        client=None,
        model: str = RAPTOR_SUMMARIZER_MODEL,
        api_key: str | None = OPENAI_API_KEY,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            client = OpenAI(api_key=api_key)
        self._client = client
        self._model = model
        self._fallback = ConcatSummarizer()

    def summarize(self, parent: RaptorNode, children: list[RaptorNode]) -> str:
        try:
            return self._complete(parent, children)
        except Exception:
            return self._fallback.summarize(parent, children)

    def _complete(self, parent: RaptorNode, children: list[RaptorNode]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": cluster_prompt(parent, children)},
            ],
        )
        return (response.choices[0].message.content or "").strip()


class BedrockClusterSummarizer(ClusterSummarizer):
    """Placeholder until Amazon Bedrock access is available."""

    def summarize(self, parent: RaptorNode, children: list[RaptorNode]) -> str:
        raise NotImplementedError(
            "Bedrock RAPTOR summarizer is not enabled yet. Use OpenAIClusterSummarizer."
        )


def get_summarizer(provider: str | None = None) -> ClusterSummarizer:
    name = (provider or RAPTOR_SUMMARIZER).lower()
    if name in {"concat", "heuristic"}:
        return ConcatSummarizer()
    if name == "openai":
        return OpenAIClusterSummarizer()
    if name == "bedrock":
        return BedrockClusterSummarizer()
    raise ValueError(f"Unknown RAPTOR summarizer: {name}")


def cluster_prompt(parent: RaptorNode, children: list[RaptorNode]) -> str:
    header = []
    if parent.title:
        header.append(f"Title: {parent.title}")
    if parent.provider:
        header.append(f"Provider: {parent.provider}")
    passages = []
    for child in children:
        heading = " > ".join(child.heading_path) if child.heading_path else child.section
        passages.append(f"[{heading}]\n{child.text.strip()}")
    return "\n".join(header + ["", *passages]).strip()


def fill_summaries(
    nodes: list[RaptorNode],
    summarizer: ClusterSummarizer | None = None,
) -> list[RaptorNode]:
    """Write summary text onto parent nodes from their children."""
    backend = summarizer or get_summarizer()
    updated = [node.model_copy(deep=True) for node in nodes]
    by_id = {node.id: node for node in updated}
    for node in updated:
        if node.node_type != "summary":
            continue
        children = [by_id[child_id] for child_id in node.children if child_id in by_id]
        node.text = backend.summarize(node, children)
    return updated


def _embed_text(node: RaptorNode) -> str:
    lines = []
    if node.title:
        lines.append(f"Title: {node.title}")
    if node.provider:
        lines.append(f"Provider: {node.provider}")
    lines.append(f"Section: {node.section or 'Summary'}")
    lines.append("")
    lines.append(node.text)
    return "\n".join(lines)


def build_raptor_tree(
    leaves: list[RaptorNode],
    vectors: list[list[float]],
    *,
    summarizer: ClusterSummarizer | None = None,
    embedder: Embedder | None = None,
    max_levels: int = RAPTOR_MAX_LEVELS,
    k: int | None = None,
    by_document: bool = True,
) -> RaptorTree:
    """Cluster leaves, summarize parents, and optionally stack higher levels."""
    backend = summarizer or get_summarizer()
    tree = RaptorTree()
    if not leaves:
        return tree

    current = [leaf.model_copy(deep=True) for leaf in leaves]
    current_vectors = list(vectors)

    for _ in range(max(1, max_levels)):
        if len(current) <= 1:
            for node in current:
                tree.add(node)
            return tree
        level_k = choose_cluster_count(len(current)) if k is None else k
        level_k = max(1, min(level_k, len(current) - 1))
        leveled = fill_summaries(
            build_tree_level(current, current_vectors, k=level_k, by_document=by_document),
            backend,
        )
        current_ids = {node.id for node in current}
        parents = [node for node in leveled if node.id not in current_ids]
        for node in leveled:
            tree.add(node)
        if len(parents) <= 1 or embedder is None:
            return tree
        current = parents
        current_vectors = embedder.embed([_embed_text(node) for node in current])
    return tree
