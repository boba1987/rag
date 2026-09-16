from typing import Literal

from pydantic import BaseModel, Field

ContentType = Literal["article", "review", "provider"]


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
