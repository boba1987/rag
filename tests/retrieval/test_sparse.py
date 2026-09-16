from app.models.schemas import Chunk, RetrievalFilters
from app.retrieval.sparse import SparseRetriever, tokenize


def _chunk(**overrides) -> Chunk:
    data = {
        "id": "provider_8015_integrations_01",
        "document_id": "8015",
        "content_type": "provider",
        "provider": "RingCentral",
        "title": "RingCentral",
        "section": "Analytics Integrations",
        "heading_path": ["RingCentral", "Analytics Integrations"],
        "text": "RingCentral integrates with Salesforce, Theta Lake, and Gong.",
    }
    data.update(overrides)
    return Chunk(**data)


_SALESFORCE = _chunk()
_PRICING = _chunk(
    id="provider_8019_core_01",
    document_id="8019",
    provider="Nextiva",
    title="Nextiva",
    section="Nextiva Core ($15-$23/user per month)",
    heading_path=["Nextiva", "Nextiva Core ($15-$23/user per month)"],
    text="Nextiva Core includes 100 SMS text messages per user per month.",
)


def test_tokenize_keeps_lexical_terms() -> None:
    assert "salesforce" in tokenize("Does RingCentral integrate with Salesforce?")
    assert "$15" in tokenize("Core is $15 per user")


def test_salesforce_query_ranks_integration_chunk_first() -> None:
    retriever = SparseRetriever(chunks=[_PRICING, _SALESFORCE], top_k=2)
    results = retriever.search("Does RingCentral integrate with Salesforce?")
    assert results[0].document_id == "8015"
    assert "Salesforce" in results[0].text
    if len(results) > 1:
        assert results[0].score > results[1].score


def test_provider_filter_keeps_nextiva_only() -> None:
    retriever = SparseRetriever(chunks=[_PRICING, _SALESFORCE], top_k=5)
    results = retriever.search(
        "Salesforce SMS pricing",
        filters=RetrievalFilters(provider="Nextiva"),
    )
    assert results
    assert {chunk.provider for chunk in results} == {"Nextiva"}


def test_empty_query_returns_no_hits() -> None:
    retriever = SparseRetriever(chunks=[_SALESFORCE], top_k=5)
    assert retriever.search("") == []
