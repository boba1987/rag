from app.models.schemas import RetrievalFilters, RetrievedChunk
from app.query.corrective import broaden_filters, corrective_rewrite, retrieve_with_correction
from app.query.evidence import HeuristicEvidenceChecker
from app.query.extractor import QueryExtraction


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id="c1",
        document_id="8015",
        content_type="provider",
        provider="RingCentral",
        title="RingCentral",
        section="Integrations",
        heading_path=["RingCentral", "Integrations"],
        text=text,
        score=0.8,
    )


class _ScriptedRetriever:
    def __init__(self, by_query: dict[str, list[RetrievedChunk]]) -> None:
        self.by_query = by_query
        self.queries: list[str] = []
        self.filters = []

    def search(self, query: str, top_k=None, filters=None, infer: bool = False):
        self.queries.append(query)
        self.filters.append(filters)
        return self.by_query.get(query, [])


def test_sufficient_first_pass_does_not_retry() -> None:
    searcher = _ScriptedRetriever(
        {"RingCentral Salesforce": [_chunk("RingCentral supports Salesforce.")]}
    )
    result = retrieve_with_correction(
        "Does RingCentral integrate with Salesforce?",
        searcher,
        ["RingCentral Salesforce"],
        checker=HeuristicEvidenceChecker(),
    )
    assert result.retried is False
    assert result.verdict.sufficient is True
    assert searcher.queries == ["RingCentral Salesforce"]


def test_insufficient_first_pass_retrieves_again() -> None:
    searcher = _ScriptedRetriever(
        {
            "Zoom revenue": [_chunk("RingCentral supports Salesforce.")],
            "What is Zoom Phone's 2020 revenue?": [_chunk("Still no Zoom revenue figures.")],
        }
    )
    result = retrieve_with_correction(
        "What is Zoom Phone's 2020 revenue?",
        searcher,
        ["Zoom revenue"],
        checker=HeuristicEvidenceChecker(),
    )
    assert result.retried is True
    assert "What is Zoom Phone's 2020 revenue?" in searcher.queries
    assert result.retry_queries


def test_corrective_rewrite_skips_queries_already_tried() -> None:
    queries = corrective_rewrite(
        "Does RingCentral integrate with Salesforce?",
        extraction=QueryExtraction(
            kind="factual",
            providers=["RingCentral"],
            source="openai",
        ),
        first_queries=["Does RingCentral integrate with Salesforce?"],
    )
    assert "Does RingCentral integrate with Salesforce?" not in queries
    assert "RingCentral" in queries


def test_broaden_filters_drops_section() -> None:
    broadened = broaden_filters(RetrievalFilters(provider="RingCentral", section="Pricing"))
    assert broadened is not None
    assert broadened.provider == "RingCentral"
    assert broadened.section is None
