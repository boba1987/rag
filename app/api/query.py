from fastapi import APIRouter, Request, Security

from app.api.auth import API_KEY_HEADER
from app.generation.generator import get_generator
from app.models.schemas import QueryRequest, QueryResponse, RetrievalFilters, RetrievalStrategy
from app.workflows.rag_graph import run_rag_graph

router = APIRouter(tags=["API"], dependencies=[Security(API_KEY_HEADER)])


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
    return run_rag_graph(
        question,
        retriever=retriever,
        generator=generator,
        extractor=extractor,
        catalog=catalog,
        filters=filters,
        infer=infer,
        strategy=strategy,
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
