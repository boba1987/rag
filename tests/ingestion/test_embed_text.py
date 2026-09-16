from app.ingestion.embed_text import build_embedding_text
from app.models.schemas import Chunk


def test_prefixes_title_provider_section_before_body() -> None:
    chunk = Chunk(
        id="review_123_integrations_01",
        document_id="123",
        content_type="review",
        provider="RingCentral",
        title="RingCentral Review",
        section="Integrations",
        heading_path=["RingCentral Review", "Integrations"],
        text="RingCentral supports Salesforce.",
        source_url="https://example.test/review",
        updated_at="2026-04-13 10:03:33",
    )
    assert build_embedding_text(chunk) == (
        "Title: RingCentral Review\n"
        "Provider: RingCentral\n"
        "Section: Integrations\n"
        "\n"
        "RingCentral supports Salesforce."
    )


def test_omits_provider_line_when_missing() -> None:
    chunk = Chunk(
        id="article_1_intro_01",
        document_id="1",
        content_type="article",
        title="Buying Guide",
        section="Buying Guide",
        heading_path=["Buying Guide"],
        text="Compare plans first.",
    )
    text = build_embedding_text(chunk)
    assert text.startswith("Title: Buying Guide\nSection: Buying Guide\n")
    assert "Provider:" not in text
    assert "updated_at" not in text
