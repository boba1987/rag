from app.models.schemas import RetrievedChunk
from app.workflows.rag_graph import build_rag_graph, run_rag_graph
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
