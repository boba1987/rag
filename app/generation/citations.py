from app.models.schemas import RetrievedChunk, Source


def sources_from_chunks(chunks: list[RetrievedChunk]) -> list[Source]:
    """Unique title/section/url citations in retrieval order."""
    sources: list[Source] = []
    seen: set[tuple[str, str, str | None]] = set()
    for chunk in chunks:
        key = (chunk.title, chunk.section, chunk.source_url)
        if key in seen:
            continue
        seen.add(key)
        sources.append(Source(title=chunk.title, section=chunk.section, url=chunk.source_url))
    return sources
