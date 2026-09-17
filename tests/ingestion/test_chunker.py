from app.ingestion.chunker import (
    FixedSizeChunker,
    ParentChildChunker,
    StructureAwareChunker,
    count_tokens,
    get_chunker,
)
from app.models.schemas import NormalizedDocument, Section


def _document(*sections: Section) -> NormalizedDocument:
    return NormalizedDocument(
        id="8019",
        title="Nextiva",
        content_type="provider",
        provider="Nextiva",
        url="http://localhost/review/nextiva-2/",
        updated_at="2026-09-02 06:31:16",
        sections=list(sections),
    )


def test_keeps_short_section_as_one_chunk() -> None:
    document = _document(
        Section(heading="Pricing", heading_path=["Nextiva", "Pricing"], text="Three plans start at $15."),
        Section(heading="Support", heading_path=["Nextiva", "Support"], text="24/7 phone and chat."),
    )
    chunks = StructureAwareChunker().chunk(document)
    assert len(chunks) == 2
    assert chunks[0].id == "provider_8019_pricing_01"
    assert chunks[0].section == "Pricing"
    assert chunks[0].heading_path == ["Nextiva", "Pricing"]
    assert chunks[0].source_url == document.url
    assert chunks[0].document_id == "8019"
    assert chunks[0].updated_at == "2026-09-02 06:31:16"
    assert chunks[1].section == "Support"


def test_splits_long_section_under_target_max() -> None:
    chunker = StructureAwareChunker(target_min=50, target_max=80, hard_max=120, overlap=10)
    text = " ".join(f"word{i}." for i in range(200))
    document = _document(Section(heading="Long", heading_path=["Nextiva", "Long"], text=text))
    chunks = chunker.chunk(document)
    assert len(chunks) > 1
    assert all(count_tokens(chunk.text) <= chunker.hard_max for chunk in chunks)
    assert all(chunk.section == "Long" for chunk in chunks)
    assert chunks[0].id.endswith("_01")
    assert chunks[1].id.endswith("_02")


def test_review_is_single_chunk_with_provider() -> None:
    document = NormalizedDocument(
        id="106948",
        title="We've had a positive experience with...",
        content_type="review",
        provider="Nextiva",
        url="https://getvoip.dev/?user-reviews=example",
        sections=[
            Section(
                heading="We've had a positive experience with...",
                heading_path=["We've had a positive experience with..."],
                text="Reliable phones.\n\nHelpful support.",
            )
        ],
    )
    chunks = StructureAwareChunker().chunk(document)
    assert len(chunks) == 1
    assert chunks[0].content_type == "review"
    assert chunks[0].provider == "Nextiva"
    assert "Reliable phones." in chunks[0].text


def test_fixed_size_keeps_short_document_as_one_chunk() -> None:
    document = _document(
        Section(heading="Pricing", heading_path=["Nextiva", "Pricing"], text="Three plans start at $15."),
        Section(heading="Support", heading_path=["Nextiva", "Support"], text="24/7 phone and chat."),
    )
    chunks = FixedSizeChunker(size=50, overlap=5).chunk(document)
    assert len(chunks) == 1
    assert chunks[0].id == "provider_8019_fixed_01"
    assert "Three plans start at $15." in chunks[0].text
    assert "24/7 phone and chat." in chunks[0].text
    assert chunks[0].section == "Nextiva"


def test_fixed_size_ignores_headings_and_windows_tokens() -> None:
    chunker = FixedSizeChunker(size=20, overlap=5)
    text = " ".join(f"word{i}" for i in range(50))
    document = _document(
        Section(heading="A", heading_path=["Nextiva", "A"], text=text),
        Section(heading="B", heading_path=["Nextiva", "B"], text="tail"),
    )
    chunks = chunker.chunk(document)
    assert len(chunks) > 1
    assert all(count_tokens(chunk.text) <= 20 for chunk in chunks)
    assert chunks[0].id.endswith("_fixed_01")
    assert chunks[1].id.endswith("_fixed_02")
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-5:] == second_words[:5]


def test_get_chunker_selects_implemented_chunkers() -> None:
    assert isinstance(get_chunker("fixed_size"), FixedSizeChunker)
    assert isinstance(get_chunker("structure_aware"), StructureAwareChunker)
    assert isinstance(get_chunker("parent_child"), ParentChildChunker)


def test_parent_child_keeps_section_as_parent() -> None:
    document = _document(
        Section(heading="Pricing", heading_path=["Nextiva", "Pricing"], text="Three plans start at $15."),
    )
    chunker = ParentChildChunker(child_size=50, overlap=5)
    children = chunker.chunk(document)
    parents = chunker.parents(document)
    assert len(children) == 1
    assert len(parents) == 1
    assert children[0].id == "provider_8019_pricing_child_01"
    assert children[0].parent_id == "provider_8019_pricing_parent"
    assert parents[0].id == children[0].parent_id
    assert parents[0].text == "Three plans start at $15."
    assert parents[0].parent_id is None


def test_parent_child_splits_long_section_into_children() -> None:
    text = " ".join(f"word{i}" for i in range(40))
    document = _document(Section(heading="Long", heading_path=["Nextiva", "Long"], text=text))
    chunker = ParentChildChunker(child_size=15, overlap=5)
    children = chunker.chunk(document)
    parents = chunker.parents(document)
    assert len(children) > 1
    assert len(parents) == 1
    assert all(chunk.parent_id == parents[0].id for chunk in children)
    assert all(count_tokens(chunk.text) <= 15 for chunk in children)
    assert count_tokens(parents[0].text) == 40
    assert children[0].section == "Long"
