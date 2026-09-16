from app.models.schemas import RetrievedChunk


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Turn retrieved chunks into numbered passages for the generator."""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        header = f"[{index}] Title: {chunk.title} | Section: {chunk.section}"
        if chunk.source_url:
            header += f" | URL: {chunk.source_url}"
        parts.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n".join(parts)
