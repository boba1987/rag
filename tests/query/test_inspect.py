from app.models.schemas import RetrievedChunk
from app.query.evidence import ABSTAIN_MESSAGE
from app.query.inspect import (
    inspect_evidence,
    inspect_preprocess,
    inspect_raptor,
    write_evidence_inspect,
    write_preprocess_inspect,
    write_raptor_inspect,
)
from tests.query.fakes import DEFAULT_SCRIPTED


def test_inspect_preprocess_covers_default_questions(tmp_path) -> None:
    rows = inspect_preprocess(extractor=DEFAULT_SCRIPTED)
    assert len(rows) == 5
    multi = next(row for row in rows if "pricing and Salesforce" in row["query"])
    assert multi["kind"] == "multi-hop"
    assert multi["extractor"] == "openai"
    assert len(multi["queries"]) == 4
    path = write_preprocess_inspect(path=tmp_path / "preprocess-inspect.json", extractor=DEFAULT_SCRIPTED)
    assert path.is_file()


class _SalesforceRetriever:
    def search(self, query: str, top_k=None, filters=None, infer: bool = False):
        if "Zoom" in query:
            return []
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


class _FixedGenerator:
    def generate(self, question: str, context: str) -> str:
        return "grounded"


def test_inspect_evidence_marks_abstain(tmp_path) -> None:
    rows = inspect_evidence(
        questions=["Does RingCentral integrate with Salesforce?", "What is Zoom Phone's 2020 revenue?"],
        retriever=_SalesforceRetriever(),
        generator=_FixedGenerator(),
        extractor=DEFAULT_SCRIPTED,
    )
    assert rows[0]["abstained"] is False
    assert rows[1]["abstained"] is True
    assert rows[1]["answer"] == ABSTAIN_MESSAGE
    path = write_evidence_inspect(
        path=tmp_path / "evidence-inspect.json",
        questions=["What is Zoom Phone's 2020 revenue?"],
        retriever=_SalesforceRetriever(),
        generator=_FixedGenerator(),
        extractor=DEFAULT_SCRIPTED,
    )
    assert path.is_file()


def test_inspect_raptor_echoes_strategy(tmp_path) -> None:
    class _SummaryRetriever:
        def search(self, query: str, top_k=None, filters=None, infer: bool = False):
            return [
                RetrievedChunk(
                    id="raptor_100_l1_00",
                    document_id="100",
                    content_type="review",
                    provider="RingCentral",
                    title="RingCentral Review",
                    section="Summary",
                    heading_path=["RingCentral Review", "Summary"],
                    text="RingCentral strengths include pricing; weaknesses include a learning curve.",
                    score=0.9,
                )
            ]

    rows = inspect_raptor(
        questions=["What are the main strengths and weaknesses of RingCentral?"],
        retriever=_SummaryRetriever(),
        generator=_FixedGenerator(),
        extractor=DEFAULT_SCRIPTED,
    )
    assert rows[0]["strategy"] == "raptor"
    assert rows[0]["sources"][0]["section"] == "Summary"
    path = write_raptor_inspect(
        path=tmp_path / "raptor-inspect.json",
        questions=["What are the main strengths and weaknesses of RingCentral?"],
        retriever=_SummaryRetriever(),
        generator=_FixedGenerator(),
        extractor=DEFAULT_SCRIPTED,
    )
    assert path.is_file()
