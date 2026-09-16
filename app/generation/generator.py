from __future__ import annotations

from abc import ABC, abstractmethod

from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL
from app.generation.citations import sources_from_chunks
from app.generation.context import build_context
from app.generation.prompts import SYSTEM_PROMPT, user_prompt
from app.models.schemas import GroundedAnswer, RetrievedChunk


class Generator(ABC):
    """Swappable generation backend. OpenAI now; Bedrock Claude/Nova later."""

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        raise NotImplementedError


class OpenAIGenerator(Generator):
    def __init__(
        self,
        api_key: str | None = OPENAI_API_KEY,
        model: str = OPENAI_CHAT_MODEL,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, question: str, context: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(question, context)},
            ],
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class BedrockGenerator(Generator):
    """Placeholder until Amazon Bedrock access is available."""

    def generate(self, question: str, context: str) -> str:
        raise NotImplementedError(
            "Bedrock generator is not enabled yet. Use OpenAIGenerator until Bedrock access exists."
        )


def get_generator(provider: str | None = None) -> Generator:
    from app.config import GENERATOR_PROVIDER

    name = (provider or GENERATOR_PROVIDER).lower()
    if name == "openai":
        return OpenAIGenerator()
    if name == "bedrock":
        return BedrockGenerator()
    raise ValueError(f"Unknown generator provider: {name}")


def generate_grounded_answer(
    question: str,
    chunks: list[RetrievedChunk],
    generator: Generator | None = None,
) -> GroundedAnswer:
    backend = generator or get_generator()
    answer = backend.generate(question, build_context(chunks))
    return GroundedAnswer(answer=answer, sources=sources_from_chunks(chunks))
