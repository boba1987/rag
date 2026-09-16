import pytest

from app.ingestion.embedder import BedrockTitanEmbedder, OpenAIEmbedder, get_embedder


def test_openai_embedder_requires_api_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(api_key=None)


def test_bedrock_stub_is_not_enabled() -> None:
    embedder = BedrockTitanEmbedder()
    assert embedder.dimensions == 1024
    with pytest.raises(NotImplementedError, match="Bedrock Titan"):
        embedder.embed(["Nextiva pricing"])


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown embedder provider"):
        get_embedder("cohere")
