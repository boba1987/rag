from __future__ import annotations

from dataclasses import dataclass, field

from app.config import DENSE_TOP_K
from app.models.schemas import RetrievalFilters, RetrievedChunk
from app.observability.tracing import span
from app.query.catalog import QueryCatalog
from app.query.evidence import EvidenceChecker, EvidenceVerdict, check_evidence
from app.query.extractor import QueryExtraction
from app.query.rewriter import rewrite_query
from app.retrieval.fusion import reciprocal_rank_fusion


@dataclass
class CorrectiveRetrieval:
    chunks: list[RetrievedChunk]
    verdict: EvidenceVerdict
    retried: bool
    queries: list[str]
    filters: RetrievalFilters | None
    retry_queries: list[str] = field(default_factory=list)


def corrective_rewrite(
    question: str,
    extraction: QueryExtraction | None = None,
    first_queries: list[str] | None = None,
    catalog: QueryCatalog | None = None,
) -> list[str]:
    """Broader follow-up queries: original question plus provider names, minus the first attempt."""
    rewritten = rewrite_query(question, extraction=None, catalog=catalog)
    candidates = [question.strip(), rewritten]
    if extraction:
        candidates.extend(extraction.providers)
    seen = {query.lower() for query in (first_queries or [])}
    fresh = [query for query in _unique(candidates) if query and query.lower() not in seen]
    return fresh or [question.strip()]


def broaden_filters(filters: RetrievalFilters | None) -> RetrievalFilters | None:
    if filters is None or filters.section is None:
        return filters
    return filters.model_copy(update={"section": None})


def retrieve_with_correction(
    question: str,
    searcher,
    queries: list[str],
    filters: RetrievalFilters | None = None,
    checker: EvidenceChecker | None = None,
    extraction: QueryExtraction | None = None,
    catalog: QueryCatalog | None = None,
) -> CorrectiveRetrieval:
    """Retrieve once; if evidence is thin, rewrite, drop the section filter, and retrieve again."""
    chunks = _traced_search("retrieval.attempt", searcher, queries, filters)
    verdict = _traced_check(question, chunks, checker, attempt=1)
    if verdict.sufficient:
        return CorrectiveRetrieval(
            chunks=chunks,
            verdict=verdict,
            retried=False,
            queries=queries,
            filters=filters,
        )
    retry_queries = corrective_rewrite(
        question,
        extraction=extraction,
        first_queries=queries,
        catalog=catalog,
    )
    retry_filters = broaden_filters(filters)
    retry_chunks = _traced_search("retrieval.retry", searcher, retry_queries, retry_filters)
    retry_verdict = _traced_check(question, retry_chunks, checker, attempt=2)
    return CorrectiveRetrieval(
        chunks=retry_chunks,
        verdict=retry_verdict,
        retried=True,
        queries=queries,
        filters=retry_filters,
        retry_queries=retry_queries,
    )


def _traced_search(
    name: str,
    searcher,
    queries: list[str],
    filters: RetrievalFilters | None,
) -> list[RetrievedChunk]:
    with span(
        name,
        input={
            "queries": queries,
            "filters": filters.model_dump(exclude_none=True) if filters else None,
        },
    ) as search_span:
        chunks = run_searches(searcher, queries, filters)
        search_span.update(
            output={
                "chunk_ids": [chunk.id for chunk in chunks],
                "document_ids": [chunk.document_id for chunk in chunks],
                "scores": [chunk.score for chunk in chunks],
            }
        )
    return chunks


def _traced_check(
    question: str,
    chunks: list[RetrievedChunk],
    checker: EvidenceChecker | None,
    attempt: int,
) -> EvidenceVerdict:
    with span(
        "evidence.check",
        input={"query": question, "chunks": len(chunks)},
        metadata={"attempt": attempt},
    ) as check_span:
        verdict = check_evidence(question, chunks, checker=checker)
        check_span.update(output=verdict.model_dump())
    return verdict


def run_searches(searcher, queries: list[str], filters: RetrievalFilters | None):
    if len(queries) == 1:
        return searcher.search(queries[0], filters=filters)
    rankings = [searcher.search(query, filters=filters) for query in queries]
    return reciprocal_rank_fusion(rankings)[:DENSE_TOP_K]


def _unique(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for query in queries:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(query)
    return ordered
