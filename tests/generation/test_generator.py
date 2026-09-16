import pytest

from app.generation.generator import (
    BedrockGenerator,
    OpenAIGenerator,
    generate_grounded_answer,
    get_generator,
)
from app.models.schemas import RetrievedChunk


class _FakeGenerator:
    def generate(self, question: str, context: str) -> str:
        return f"grounded:{question}:{context[:20]}"


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        id="provider_8015_integrations_01",
        document_id="8015",
        content_type="provider",
        provider="RingCentral",
        title="RingCentral Review",
        section="Integrations",
        heading_path=["RingCentral Review", "Integrations"],
        text="RingCentral supports Salesforce.",
        source_url="https://example.test/ringcentral",
        score=0.91,
    )


def test_openai_generator_requires_api_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIGenerator(api_key=None)


def test_bedrock_stub_is_not_enabled() -> None:
    with pytest.raises(NotImplementedError, match="Bedrock generator"):
        BedrockGenerator().generate("Does RingCentral integrate with Salesforce?", "context")


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown generator provider"):
        get_generator("cohere")


def test_grounded_answer_uses_context_and_citations() -> None:
    result = generate_grounded_answer(
        "Does RingCentral integrate with Salesforce?",
        [_chunk()],
        generator=_FakeGenerator(),
    )
    assert result.answer.startswith("grounded:Does RingCentral")
    assert result.sources[0].title == "RingCentral Review"
    assert result.sources[0].section == "Integrations"
    assert result.sources[0].url == "https://example.test/ringcentral"
