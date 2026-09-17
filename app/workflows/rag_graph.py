from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from app.generation.generator import generate_grounded_answer
from app.models.schemas import (
    EvidenceInfo,
    GroundedAnswer,
    QueryPreprocess,
    QueryResponse,
    RetrievalFilters,
    RetrievalInfo,
    RetrievalStrategy,
)
from app.query.corrective import CorrectiveRetrieval, retrieve_with_correction
from app.query.decomposer import expand_queries
from app.query.evidence import ABSTAIN_MESSAGE
from app.query.extractor import understand_query
from app.query.rewriter import rewrite_query
from app.retrieval import retriever_for
from app.retrieval.filters import infer_filters, merge_filters


class RagState(TypedDict):
    question: str
    strategy: RetrievalStrategy
    infer: bool
    filters: RetrievalFilters | None
    retriever: NotRequired[Any]
    generator: NotRequired[Any]
    extractor: NotRequired[Any]
    catalog: NotRequired[Any]
    extraction: NotRequired[Any]
    queries: NotRequired[list[str]]
    applied: NotRequired[RetrievalFilters | None]
    corrected: NotRequired[CorrectiveRetrieval]
    response: NotRequired[QueryResponse]


def preprocess(state: RagState) -> dict:
    extraction = understand_query(
        state["question"],
        extractor=state.get("extractor"),
        catalog=state.get("catalog"),
    )
    queries = expand_queries(
        state["question"],
        extraction=extraction,
        catalog=state.get("catalog"),
    )
    applied = merge_filters(
        state.get("filters"),
        infer_filters(state["question"], extraction) if state.get("infer", True) else None,
    )
    return {"extraction": extraction, "queries": queries, "applied": applied}


def retrieve(state: RagState) -> dict:
    searcher = state.get("retriever") or retriever_for(state["strategy"])
    corrected = retrieve_with_correction(
        state["question"],
        searcher,
        state["queries"],
        filters=state.get("applied"),
        extraction=state.get("extraction"),
        catalog=state.get("catalog"),
    )
    return {"corrected": corrected}


def respond(state: RagState) -> dict:
    """Commit 1: one node. Commit 2 splits generate vs abstain."""
    corrected = state["corrected"]
    extraction = state["extraction"]
    if corrected.verdict.sufficient:
        grounded = generate_grounded_answer(
            state["question"],
            corrected.chunks,
            generator=state.get("generator"),
        )
    else:
        grounded = GroundedAnswer(answer=ABSTAIN_MESSAGE, sources=[])
    return {
        "response": QueryResponse(
            answer=grounded.answer,
            sources=grounded.sources,
            retrieval=RetrievalInfo(
                strategy=state["strategy"],
                filters=corrected.filters,
                inferred=state.get("infer", True),
                preprocess=QueryPreprocess(
                    kind=extraction.kind,
                    rewritten=rewrite_query(
                        state["question"],
                        extraction=extraction,
                        catalog=state.get("catalog"),
                    ),
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
    }


def build_rag_graph():
    builder = StateGraph(RagState)
    builder.add_node("preprocess", preprocess)
    builder.add_node("retrieve", retrieve)
    builder.add_node("respond", respond)
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "retrieve")
    builder.add_edge("retrieve", "respond")
    builder.add_edge("respond", END)
    return builder.compile()


def run_rag_graph(
    question: str,
    *,
    retriever=None,
    generator=None,
    extractor=None,
    catalog=None,
    filters: RetrievalFilters | None = None,
    infer: bool = True,
    strategy: RetrievalStrategy = "dense",
) -> QueryResponse:
    result = build_rag_graph().invoke(
        {
            "question": question,
            "strategy": strategy,
            "infer": infer,
            "filters": filters,
            "retriever": retriever,
            "generator": generator,
            "extractor": extractor,
            "catalog": catalog,
        }
    )
    return result["response"]
