from types import SimpleNamespace

import pytest

from app.ingestion.embedder import Embedder
from app.models.schemas import Chunk
from app.raptor.summarization import (
    BedrockClusterSummarizer,
    ConcatSummarizer,
    OpenAIClusterSummarizer,
    build_raptor_tree,
    cluster_prompt,
    fill_summaries,
    get_summarizer,
)
from app.raptor.tree import RaptorNode, leaf_from_chunk


class _FixedEmbedder(Embedder):
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors
        self.texts: list[str] = []

    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [self._vectors[index % len(self._vectors)] for index in range(len(texts))]


class _ScriptedSummarizer(ConcatSummarizer):
    def summarize(self, parent: RaptorNode, children: list[RaptorNode]) -> str:
        return f"summary:{parent.id}:{','.join(child.id for child in children)}"


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


def _parent_with_children() -> list[RaptorNode]:
    pricing = leaf_from_chunk(_chunk(id="pricing", section="Pricing", text="Core is $30."))
    features = leaf_from_chunk(_chunk(id="features", section="Features", text="Includes video."))
    parent = RaptorNode(
        id="raptor_100_l1_00",
        level=1,
        node_type="summary",
        title="RingCentral Review",
        provider="RingCentral",
        children=["pricing", "features"],
    )
    pricing.parent_id = parent.id
    features.parent_id = parent.id
    return [parent, pricing, features]


def test_concat_summarizer_joins_child_headings() -> None:
    parent, pricing, features = _parent_with_children()
    text = ConcatSummarizer().summarize(parent, [pricing, features])
    assert "Pricing: Core is $30." in text
    assert "Features: Includes video." in text


def test_fill_summaries_writes_parent_text() -> None:
    nodes = fill_summaries(_parent_with_children(), ConcatSummarizer())
    parent = next(node for node in nodes if node.node_type == "summary")
    assert "Core is $30." in parent.text
    assert "Includes video." in parent.text


def test_cluster_prompt_includes_title_and_passages() -> None:
    parent, pricing, features = _parent_with_children()
    prompt = cluster_prompt(parent, [pricing, features])
    assert "Title: RingCentral Review" in prompt
    assert "Provider: RingCentral" in prompt
    assert "[RingCentral Review > Pricing]" in prompt
    assert "Core is $30." in prompt


def test_openai_summarizer_uses_chat_completion() -> None:
    class _Completions:
        def __init__(self) -> None:
            self.model = None
            self.messages = None

        def create(self, **kwargs):
            self.model = kwargs["model"]
            self.messages = kwargs["messages"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="  Core costs $30.  "))]
            )

    completions = _Completions()

    class _Client:
        chat = SimpleNamespace(completions=completions)

    parent, pricing, features = _parent_with_children()
    text = OpenAIClusterSummarizer(client=_Client()).summarize(parent, [pricing, features])
    assert text == "Core costs $30."
    assert completions.model == "gpt-4.1-nano"
    assert completions.messages[0]["role"] == "system"


def test_openai_summarizer_falls_back_when_llm_fails() -> None:
    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("down")

    class _Boom:
        chat = SimpleNamespace(completions=_Completions())

    parent, pricing, features = _parent_with_children()
    text = OpenAIClusterSummarizer(client=_Boom()).summarize(parent, [pricing, features])
    assert "Pricing: Core is $30." in text


def test_openai_summarizer_requires_api_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIClusterSummarizer(api_key=None)


def test_bedrock_stub_is_not_enabled() -> None:
    parent, pricing, features = _parent_with_children()
    with pytest.raises(NotImplementedError, match="Bedrock RAPTOR summarizer"):
        BedrockClusterSummarizer().summarize(parent, [pricing, features])


def test_get_summarizer_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown RAPTOR summarizer"):
        get_summarizer("claude")


def test_build_raptor_tree_summarizes_one_level_without_embedder() -> None:
    leaves = [
        leaf_from_chunk(_chunk(id="pricing", section="Pricing", text="price")),
        leaf_from_chunk(_chunk(id="features", section="Features", text="features")),
        leaf_from_chunk(_chunk(id="support", section="Support", text="support")),
        leaf_from_chunk(_chunk(id="reviews", section="Reviews", text="reviews")),
    ]
    vectors = [[1.0, 0.0], [0.95, 0.1], [0.0, 1.0], [0.1, 0.95]]
    tree = build_raptor_tree(leaves, vectors, summarizer=_ScriptedSummarizer(), k=2)

    parents = [node for node in tree.nodes.values() if node.node_type == "summary"]
    assert len(parents) == 2
    assert all(parent.text.startswith("summary:") for parent in parents)
    assert all(parent.parent_id is None for parent in parents)
    assert {node.id for node in tree.leaves()} == {"pricing", "features", "support", "reviews"}


def test_build_raptor_tree_stacks_a_higher_level_when_embedder_is_present() -> None:
    leaves = [
        leaf_from_chunk(_chunk(id="pricing", section="Pricing", text="price")),
        leaf_from_chunk(_chunk(id="features", section="Features", text="features")),
        leaf_from_chunk(_chunk(id="support", section="Support", text="support")),
        leaf_from_chunk(_chunk(id="reviews", section="Reviews", text="reviews")),
    ]
    leaf_vectors = [[1.0, 0.0], [0.95, 0.1], [0.0, 1.0], [0.1, 0.95]]
    embedder = _FixedEmbedder([[1.0, 0.0], [0.9, 0.1]])
    tree = build_raptor_tree(
        leaves,
        leaf_vectors,
        summarizer=_ScriptedSummarizer(),
        embedder=embedder,
        max_levels=2,
        k=2,
    )

    levels = {node.level for node in tree.nodes.values()}
    assert levels == {0, 1, 2}
    roots = tree.roots()
    assert len(roots) == 1
    assert roots[0].level == 2
    assert roots[0].text.startswith("summary:")
    assert embedder.texts
    assert len(tree.children_of(roots[0].id)) == 2
