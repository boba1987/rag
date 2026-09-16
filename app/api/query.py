from fastapi import APIRouter, Request

from app.generation.generator import generate_grounded_answer, get_generator
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievalFilters,
    RetrievalInfo,
    RetrievalStrategy,
)
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
) -> QueryResponse:
    searcher = retriever or retriever_for(strategy)
    applied = merge_filters(filters, infer_filters(question) if infer else None)
    chunks = searcher.search(question, filters=applied)
    grounded = generate_grounded_answer(question, chunks, generator=generator)
    return QueryResponse(
        answer=grounded.answer,
        sources=grounded.sources,
        retrieval=RetrievalInfo(strategy=strategy, filters=applied, inferred=infer),
    )


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    retriever = getattr(request.app.state, "retriever", None)
    generator = getattr(request.app.state, "generator", None)
    return answer_query(
        body.query,
        retriever=retriever,
        generator=generator or get_generator(),
        filters=body.filters,
        infer=body.infer,
        strategy=body.strategy,
    )
