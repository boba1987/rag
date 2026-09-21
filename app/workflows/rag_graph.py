from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import GENERATOR_PROVIDER, OPENAI_CHAT_MODEL
from app.generation.context import build_context
from app.generation.generator import generate_grounded_answer
from app.generation.prompts import SYSTEM_PROMPT, user_prompt
from app.models.schemas import (
    EvidenceInfo,
    GroundedAnswer,
    QueryPreprocess,
    QueryResponse,
    RetrievalFilters,
    RetrievalInfo,
    RetrievalStrategy,
)
from app.observability.tracing import get_tracer, span
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
    with span("query.classify", input={"query": state["question"]}) as classify:
        extraction = understand_query(
            state["question"],
            extractor=state.get("extractor"),
            catalog=state.get("catalog"),
        )
        classify.update(
            output={
                "kind": extraction.kind,
                "providers": extraction.providers,
                "topics": extraction.topics,
                "extractor": extraction.source,
            }
        )
    with span("query.rewrite", input={"query": state["question"]}) as rewrite:
        queries = expand_queries(
            state["question"],
            extraction=extraction,
            catalog=state.get("catalog"),
        )
        rewritten = rewrite_query(
            state["question"],
            extraction=extraction,
            catalog=state.get("catalog"),
        )
        rewrite.update(output={"rewritten": rewritten, "queries": queries})
    applied = merge_filters(
        state.get("filters"),
        infer_filters(state["question"], extraction) if state.get("infer", True) else None,
    )
    return {"extraction": extraction, "queries": queries, "applied": applied}


def retrieve(state: RagState) -> dict:
    searcher = state.get("retriever") or retriever_for(state["strategy"])
    with span(
        "retrieval",
        input={"queries": state["queries"], "strategy": state["strategy"]},
        metadata={"strategy": state["strategy"]},
    ) as retrieval_span:
        corrected = retrieve_with_correction(
            state["question"],
            searcher,
            state["queries"],
            filters=state.get("applied"),
            extraction=state.get("extraction"),
            catalog=state.get("catalog"),
        )
        retrieval_span.update(
            output={
                "chunk_ids": [chunk.id for chunk in corrected.chunks],
                "document_ids": [chunk.document_id for chunk in corrected.chunks],
                "scores": [chunk.score for chunk in corrected.chunks],
                "retried": corrected.retried,
                "retry_queries": corrected.retry_queries,
                "sufficient": corrected.verdict.sufficient,
            }
        )
    return {"corrected": corrected}


def route_after_retrieve(state: RagState) -> str:
    if state["corrected"].verdict.sufficient:
        return "generate"
    return "abstain"


def generate(state: RagState) -> dict:
    corrected = state["corrected"]
    context = build_context(corrected.chunks)
    with span(
        "generation",
        as_type="generation",
        input={"question": state["question"], "context": context, "system": SYSTEM_PROMPT},
        metadata={"provider": GENERATOR_PROVIDER},
        model=OPENAI_CHAT_MODEL if GENERATOR_PROVIDER == "openai" else GENERATOR_PROVIDER,
    ) as generation_span:
        generation_span.update(metadata={"prompt": user_prompt(state["question"], context)})
        grounded = generate_grounded_answer(
            state["question"],
            corrected.chunks,
            generator=state.get("generator"),
        )
        generation_span.update(output={"answer": grounded.answer, "sources": [s.model_dump() for s in grounded.sources]})
    return {"response": _query_response(state, grounded)}


def abstain(state: RagState) -> dict:
    with span("abstain", input={"query": state["question"]}) as abstain_span:
        grounded = GroundedAnswer(answer=ABSTAIN_MESSAGE, sources=[])
        abstain_span.update(output={"answer": grounded.answer, "reason": state["corrected"].verdict.reason})
    return {"response": _query_response(state, grounded)}


def _query_response(state: RagState, grounded: GroundedAnswer) -> QueryResponse:
    corrected = state["corrected"]
    extraction = state["extraction"]
    return QueryResponse(
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


def build_rag_graph():
    builder = StateGraph(RagState)
    builder.add_node("preprocess", preprocess)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("abstain", abstain)
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "abstain": "abstain"},
    )
    builder.add_edge("generate", END)
    builder.add_edge("abstain", END)
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
    tracer = get_tracer()
    with span(
        "rag.query",
        input={"query": question, "strategy": strategy, "infer": infer},
        metadata={"strategy": strategy},
    ) as root:
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
        response = result["response"]
        root.update(
            output={
                "answer": response.answer,
                "strategy": response.retrieval.strategy,
                "sources": [source.model_dump() for source in response.sources],
                "evidence": response.retrieval.evidence.model_dump() if response.retrieval.evidence else None,
            }
        )
    tracer.flush()
    return response
