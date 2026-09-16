from app.ingestion.chunker import StructureAwareChunker, count_tokens
from app.models.schemas import NormalizedDocument, Section


def _document(*sections: Section) -> NormalizedDocument:
    return NormalizedDocument(
        id="8019",
        title="Nextiva",
        content_type="provider",
        provider="Nextiva",
        url="http://localhost/review/nextiva-2/",
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
