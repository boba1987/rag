from typing import Literal

from pydantic import BaseModel, Field

ContentType = Literal["article", "review", "provider"]
RetrievalStrategy = Literal["dense", "sparse", "hybrid", "rerank"]
EvalCategory = Literal[
    "factual",
    "pricing",
    "features",
    "integrations",
    "comparison",
    "recommendation",
    "multi-hop",
    "ambiguous",
    "unanswerable",
]


class RawPost(BaseModel):
    """Common shape for a WordPress fixture record before HTML normalization."""

    id: str
    title: str
    slug: str
    html: str
    content_type: ContentType
    provider: str | None = None
    url: str | None = None
    published_at: str | None = None
    updated_at: str | None = None


class Section(BaseModel):
    heading: str
    heading_path: list[str] = Field(min_length=1)
    text: str


class NormalizedDocument(BaseModel):
    id: str
    title: str
    content_type: ContentType
    provider: str | None = None
    url: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    sections: list[Section] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str
    document_id: str
    content_type: ContentType
    provider: str | None = None
    title: str
    section: str
    heading_path: list[str] = Field(min_length=1)
    text: str
    source_url: str | None = None
    updated_at: str | None = None
    parent_id: str | None = None


class RetrievedChunk(Chunk):
    """A stored chunk plus its dense similarity score."""

    score: float


class Source(BaseModel):
    title: str
    section: str
    url: str | None = None


class GroundedAnswer(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)


class RetrievalFilters(BaseModel):
    provider: str | None = None
    section: str | None = None
    content_type: ContentType | None = None
    document_id: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: RetrievalFilters | None = None
    infer: bool = True
    strategy: RetrievalStrategy = "dense"


class RetrievalInfo(BaseModel):
    strategy: RetrievalStrategy = "dense"
    filters: RetrievalFilters | None = None
    inferred: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    retrieval: RetrievalInfo = Field(default_factory=RetrievalInfo)


class EvalCase(BaseModel):
    id: str
    question: str
    expected_documents: list[str] = Field(default_factory=list)
    expected_answer: str | None = None
    category: EvalCategory
