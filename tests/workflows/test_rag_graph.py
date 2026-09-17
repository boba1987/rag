from types import SimpleNamespace

from app.models.schemas import RetrievedChunk
from app.query.evidence import ABSTAIN_MESSAGE
from app.workflows.rag_graph import build_rag_graph, route_after_retrieve, run_rag_graph
from tests.query.fakes import DEFAULT_SCRIPTED


class _FakeRetriever:
    def search(self, query: str, top_k=None, filters=None, infer: bool = False):
        return [
            RetrievedChunk(
                id="c1",
                document_id="8015",
                content_type="provider",
                provider="RingCentral",
                title="RingCentral",
                section="Integrations",
                heading_path=["RingCentral", "Integrations"],
                text="RingCentral supports Salesforce.",
                score=0.9,
            )
        ]


class _FakeGenerator:
    def generate(self, question: str, context: str) -> str:
        return "Yes. RingCentral supports Salesforce."


def test_graph_compiles() -> None:
    graph = build_rag_graph()
    assert graph is not None


def test_graph_runs_existing_modules() -> None:
    result = run_rag_graph(
        "Does RingCentral integrate with Salesforce?",
        retriever=_FakeRetriever(),
        generator=_FakeGenerator(),
        extractor=DEFAULT_SCRIPTED,
        infer=False,
    )
    assert result.answer == "Yes. RingCentral supports Salesforce."
    assert result.retrieval.preprocess is not None
    assert result.retrieval.preprocess.kind == "factual"
    assert result.retrieval.evidence is not None
    assert result.retrieval.evidence.sufficient is True


class _EmptyRetriever:
    def search(self, query: str, top_k=None, filters=None, infer: bool = False):
        return []


class _BoomGenerator:
    def generate(self, question: str, context: str) -> str:
        raise AssertionError("abstain path must not generate")


def test_graph_abstains_without_calling_generator() -> None:
    result = run_rag_graph(
        "What is Zoom Phone's 2020 revenue?",
        retriever=_EmptyRetriever(),
        generator=_BoomGenerator(),
        extractor=DEFAULT_SCRIPTED,
        infer=False,
    )
    assert result.answer == ABSTAIN_MESSAGE
    assert result.sources == []
    assert result.retrieval.evidence is not None
    assert result.retrieval.evidence.sufficient is False


def test_route_after_retrieve_follows_evidence() -> None:
    assert (
        route_after_retrieve({"corrected": SimpleNamespace(verdict=SimpleNamespace(sufficient=True))})
        == "generate"
    )
    assert (
        route_after_retrieve({"corrected": SimpleNamespace(verdict=SimpleNamespace(sufficient=False))})
        == "abstain"
    )
