from app.models.schemas import RawPost


def fetch_post(post_id: int) -> RawPost:
    """Read a WordPress post from MySQL. Not implemented in RAG-001."""
    raise NotImplementedError(
        f"MySQL WordPress reader is not implemented yet (post_id={post_id}). "
        "Use JSON fixtures via app.ingestion.loader."
    )
