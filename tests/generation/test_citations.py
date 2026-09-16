from app.generation.citations import sources_from_chunks
from app.models.schemas import RetrievedChunk, Source


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
        "score": 0.91,
    }
    data.update(overrides)
    return RetrievedChunk(**data)


def test_returns_title_section_url_in_order() -> None:
    sources = sources_from_chunks(
        [
            _chunk(),
            _chunk(
                id="provider_8015_pricing_01",
                section="Pricing",
                heading_path=["RingCentral Review", "Pricing"],
                text="Plans start at $20.",
                score=0.8,
            ),
        ]
    )
    assert sources == [
        Source(title="RingCentral Review", section="Integrations", url="https://example.test/ringcentral"),
        Source(title="RingCentral Review", section="Pricing", url="https://example.test/ringcentral"),
    ]


def test_deduplicates_same_title_section_url() -> None:
    sources = sources_from_chunks([_chunk(), _chunk(id="provider_8015_integrations_02", score=0.7)])
    assert sources == [
        Source(title="RingCentral Review", section="Integrations", url="https://example.test/ringcentral")
    ]
