from fastapi import APIRouter, Request

from app.config import DENSE_TOP_K
from app.generation.generator import generate_grounded_answer, get_generator
from app.models.schemas import (
    QueryPreprocess,
    QueryRequest,
    QueryResponse,
    RetrievalFilters,
    RetrievalInfo,
    RetrievalStrategy,
)
from app.query.decomposer import expand_queries
from app.query.extractor import understand_query
from app.query.rewriter import rewrite_query
from app.retrieval import retriever_for
from app.retrieval.filters import infer_filters, merge_filters
from app.retrieval.fusion import reciprocal_rank_fusion

router = APIRouter()


def answer_query(
    question: str,
    retriever=None,
    generator=None,
    filters: RetrievalFilters | None = None,
    infer: bool = True,
    strategy: RetrievalStrategy = "dense",
    extractor=None,
    catalog=None,
) -> QueryResponse:
    extraction = understand_query(question, extractor=extractor, catalog=catalog)
    queries = expand_queries(question, extraction=extraction, catalog=catalog)
    searcher = retriever or retriever_for(strategy)
    applied = merge_filters(filters, infer_filters(question, extraction) if infer else None)
    chunks = _retrieve(searcher, queries, applied)
    grounded = generate_grounded_answer(question, chunks, generator=generator)
    return QueryResponse(
        answer=grounded.answer,
        sources=grounded.sources,
        retrieval=RetrievalInfo(
            strategy=strategy,
            filters=applied,
            inferred=infer,
            preprocess=QueryPreprocess(
                kind=extraction.kind,
                rewritten=rewrite_query(question, extraction=extraction, catalog=catalog),
                queries=queries,
                providers=extraction.providers,
                topics=extraction.topics,
                extractor=extraction.source,
            ),
        ),
    )


def _retrieve(searcher, queries: list[str], filters: RetrievalFilters | None):
    if len(queries) == 1:
        return searcher.search(queries[0], filters=filters)
    rankings = [searcher.search(query, filters=filters) for query in queries]
    return reciprocal_rank_fusion(rankings)[:DENSE_TOP_K]


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    retriever = getattr(request.app.state, "retriever", None)
    generator = getattr(request.app.state, "generator", None)
    extractor = getattr(request.app.state, "extractor", None)
    return answer_query(
        body.query,
        retriever=retriever,
        generator=generator or get_generator(),
        filters=body.filters,
        infer=body.infer,
        strategy=body.strategy,
        extractor=extractor,
    )
