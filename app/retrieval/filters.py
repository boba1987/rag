from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from app.models.schemas import RetrievalFilters


def build_qdrant_filter(filters: RetrievalFilters | None) -> Filter | None:
    """AND together payload matches. Empty filters search the whole collection."""
    if filters is None:
        return None
    must: list[FieldCondition] = []
    if filters.provider:
        must.append(FieldCondition(key="provider", match=MatchValue(value=filters.provider)))
    if filters.section:
        must.append(FieldCondition(key="section", match=MatchValue(value=filters.section)))
    if filters.content_type:
        must.append(FieldCondition(key="content_type", match=MatchValue(value=filters.content_type)))
    if filters.document_id:
        must.append(FieldCondition(key="document_id", match=MatchValue(value=filters.document_id)))
    if not must:
        return None
    return Filter(must=must)
