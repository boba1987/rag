from app.models.schemas import Chunk


def build_embedding_text(chunk: Chunk) -> str:
    """Prefix chunk body with title, provider, and section for dense retrieval."""
    lines = [f"Title: {chunk.title}"]
    if chunk.provider:
        lines.append(f"Provider: {chunk.provider}")
    lines.append(f"Section: {chunk.section}")
    lines.append("")
    lines.append(chunk.text)
    return "\n".join(lines)
