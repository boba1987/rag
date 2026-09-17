from app.models.schemas import RetrievedChunk
from app.retrieval.chunker_compare import compare_chunkers, pack_hits


def _chunk(chunk_id: str, section: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id="8015",
        content_type="provider",
        provider="RingCentral",
        title="RingCentral",
        section=section,
        heading_path=["RingCentral", section],
        text="RingCentral supports Salesforce.",
        score=0.8,
    )


class _FixedRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results

    def search(self, query: str, top_k: int | None = None, filters=None, infer: bool = False):
        return self._results


def test_compare_chunkers_keeps_strategies_separate() -> None:
    structured = [_chunk("s1", "Integrations")]
    fixed = [_chunk("f1", "RingCentral")]
    compared = compare_chunkers(
        "Salesforce",
        {
            "structure_aware": _FixedRetriever(structured),
            "fixed_size": _FixedRetriever(fixed),
        },
    )
    assert compared["structure_aware"] == pack_hits(structured)
    assert compared["fixed_size"] == pack_hits(fixed)
    assert compared["structure_aware"][0]["section"] == "Integrations"
