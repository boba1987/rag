from fastapi import APIRouter, Request

from app.generation.generator import generate_grounded_answer, get_generator
from app.models.schemas import QueryRequest, QueryResponse, RetrievalInfo
from app.retrieval.dense import DenseRetriever

router = APIRouter()


def answer_query(
    question: str,
    retriever: DenseRetriever | None = None,
    generator=None,
) -> QueryResponse:
    searcher = retriever or DenseRetriever()
    chunks = searcher.search(question)
    grounded = generate_grounded_answer(question, chunks, generator=generator)
    return QueryResponse(
        answer=grounded.answer,
        sources=grounded.sources,
        retrieval=RetrievalInfo(strategy="dense"),
    )


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    retriever = getattr(request.app.state, "retriever", None)
    generator = getattr(request.app.state, "generator", None)
    return answer_query(
        body.query,
        retriever=retriever,
        generator=generator or get_generator(),
    )
