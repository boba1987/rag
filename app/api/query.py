from fastapi import APIRouter, Request

from app.generation.generator import generate_grounded_answer, get_generator
from app.models.schemas import (
    EvidenceInfo,
    GroundedAnswer,
    QueryPreprocess,
    QueryRequest,
    QueryResponse,
    RetrievalFilters,
    RetrievalInfo,
    RetrievalStrategy,
)
from app.query.evidence import ABSTAIN_MESSAGE
from app.query.corrective import retrieve_with_correction
from app.query.decomposer import expand_queries
from app.query.extractor import understand_query
from app.query.rewriter import rewrite_query
from app.retrieval import retriever_for
from app.retrieval.filters import infer_filters, merge_filters

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
    corrected = retrieve_with_correction(
        question,
        searcher,
        queries,
        filters=applied,
        extraction=extraction,
        catalog=catalog,
    )
    if corrected.verdict.sufficient:
        grounded = generate_grounded_answer(question, corrected.chunks, generator=generator)
    else:
        grounded = GroundedAnswer(answer=ABSTAIN_MESSAGE, sources=[])
    return QueryResponse(
        answer=grounded.answer,
        sources=grounded.sources,
        retrieval=RetrievalInfo(
            strategy=strategy,
            filters=corrected.filters,
            inferred=infer,
            preprocess=QueryPreprocess(
                kind=extraction.kind,
                rewritten=rewrite_query(question, extraction=extraction, catalog=catalog),
                queries=corrected.queries + corrected.retry_queries,
                providers=extraction.providers,
                topics=extraction.topics,
                extractor=extraction.source,
            ),
            evidence=EvidenceInfo(
                sufficient=corrected.verdict.sufficient,
                reason=corrected.verdict.reason,
                overlap=corrected.verdict.overlap,
                retried=corrected.retried,
            ),
        ),
    )


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
