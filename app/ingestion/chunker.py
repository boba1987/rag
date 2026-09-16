from abc import ABC, abstractmethod

from app.models.schemas import Chunk, NormalizedDocument


class Chunker(ABC):
    """Reusable chunking strategy. Later: FixedSize, ParentChild, RAPTOR."""

    @abstractmethod
    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        raise NotImplementedError
