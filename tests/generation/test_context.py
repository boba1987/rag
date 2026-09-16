from app.generation.context import build_context
from app.models.schemas import RetrievedChunk


def _chunk(**overrides) -> RetrievedChunk:
    data = {
        "id": "provider_8015_integrations_01",
        "document_id": "8015",
        "content_type": "provider",
        "provider": "RingCentral",
        "title": "RingCentral Review",
        "section": "Integrations",
        "heading_path": ["RingCentral Review", "Integrations"],
        "text": "RingCentral supports Salesforce.",
        "source_url": "https://example.test/ringcentral",
        "updated_at": "2026-04-13 10:03:33",
        "score": 0.91,
    }
    data.update(overrides)
    return RetrievedChunk(**data)


def test_numbers_passages_with_title_section_url() -> None:
    context = build_context(
        [
            _chunk(),
            _chunk(
                id="provider_8015_pricing_01",
                section="Pricing",
                heading_path=["RingCentral Review", "Pricing"],
                text="Plans start at $20.",
                source_url=None,
                score=0.8,
            ),
        ]
    )
    assert context.startswith(
        "[1] Title: RingCentral Review | Section: Integrations | URL: https://example.test/ringcentral\n"
        "RingCentral supports Salesforce."
    )
    assert "[2] Title: RingCentral Review | Section: Pricing\nPlans start at $20." in context
    assert "URL:" not in context.split("[2]", 1)[1]


def test_empty_chunks_yield_empty_context() -> None:
    assert build_context([]) == ""
