from typing import Literal

from pydantic import BaseModel, Field

ContentType = Literal["article", "review", "provider"]


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
