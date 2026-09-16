from __future__ import annotations

from abc import ABC, abstractmethod

from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_EMBED_DIMENSIONS, OPENAI_EMBED_MODEL


class Embedder(ABC):
    """Swappable embedding backend. OpenAI now; Bedrock Titan later."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        api_key: str | None = OPENAI_API_KEY,
        model: str = OPENAI_EMBED_MODEL,
        dimensions: int = OPENAI_EMBED_DIMENSIONS,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


class BedrockTitanEmbedder(Embedder):
    """Placeholder until Amazon Bedrock access is available."""

    @property
    def dimensions(self) -> int:
        return 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Bedrock Titan embedder is not enabled yet. Use OpenAIEmbedder until Bedrock access exists."
        )


def get_embedder(provider: str | None = None) -> Embedder:
    from app.config import EMBEDDER_PROVIDER

    name = (provider or EMBEDDER_PROVIDER).lower()
    if name == "openai":
        return OpenAIEmbedder()
    if name == "bedrock":
        return BedrockTitanEmbedder()
    raise ValueError(f"Unknown embedder provider: {name}")
