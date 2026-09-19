from types import SimpleNamespace

import pytest

from app.query.catalog import QueryCatalog
from app.query.extractor import (
    HeuristicExtractor,
    OpenAIExtractor,
    QueryExtraction,
    constrain_extraction,
    get_extractor,
    understand_query,
)


def _catalog() -> QueryCatalog:
    return QueryCatalog(
        providers=("Nextiva", "RingCentral"),
        aliases=(
            ("ring central", "RingCentral"),
            ("ringcentral", "RingCentral"),
            ("nextiva", "Nextiva"),
        ),
        headings=("Plans and Pricing", "Integrations"),
        corpus="ringcentral plans and pricing salesforce integration nextiva",
    )


def test_heuristic_fallback_only_matches_providers() -> None:
    extracted = HeuristicExtractor().extract(
        "Compare RingCentral and Nextiva pricing and Salesforce integrations.",
        catalog=_catalog(),
    )
    assert extracted.source == "heuristic"
    assert extracted.kind == "factual"
    assert extracted.providers == ["RingCentral", "Nextiva"]
    assert extracted.topics == []


def test_openai_extractor_does_not_keep_unmentioned_catalog_provider() -> None:
    class _Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"kind": "pricing", "providers": ["Nextiva"], "topics": ["pricing"]}'
                        )
                    )
                ]
            )

    extracted = OpenAIExtractor(client=_Client()).extract(
        "what is NICE CXone pricing",
        catalog=_catalog(),
    )
    assert extracted.providers == []
    assert extracted.kind == "pricing"


def test_openai_extractor_uses_json_and_constrains_to_catalog() -> None:
    class _Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"kind": "pricing", "providers": ["RingCentral", "Zoom"], "topics": ["pricing"]}'
                        )
                    )
                ]
            )

    extracted = OpenAIExtractor(client=_Client()).extract(
        "How much does RingCentral cost?",
        catalog=_catalog(),
    )
    assert extracted.source == "openai"
    assert extracted.kind == "pricing"
    assert extracted.providers == ["RingCentral"]
    assert extracted.topics == ["pricing"]


def test_openai_extractor_falls_back_when_llm_fails() -> None:
    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("down")

    class _Boom:
        chat = SimpleNamespace(completions=_Completions())

    extracted = OpenAIExtractor(client=_Boom()).extract(
        "How much does RingCentral cost?",
        catalog=_catalog(),
    )
    assert extracted.source == "heuristic"
    assert extracted.kind == "factual"


def test_constrain_drops_unknown_names() -> None:
    constrained = constrain_extraction(
        QueryExtraction(kind="factual", providers=["Zoom"], topics=["billing"], source="openai"),
        _catalog(),
    )
    assert constrained.providers == []
    assert constrained.topics == []


def test_constrain_drops_technology_term_not_in_catalog() -> None:
    constrained = constrain_extraction(
        QueryExtraction(kind="factual", providers=["VoIP"], topics=["VoIP"], source="openai"),
        _catalog(),
        "what is voip?",
    )
    assert constrained.providers == []
    assert constrained.kind == "factual"


def test_constrain_does_not_substitute_a_catalog_provider() -> None:
    constrained = constrain_extraction(
        QueryExtraction(kind="pricing", providers=["Nextiva"], topics=["pricing"], source="openai"),
        _catalog(),
        "what is NICE CXone pricing",
    )
    assert constrained.providers == []
    assert constrained.kind == "pricing"


def test_understand_query_uses_injected_extractor() -> None:
    extracted = understand_query(
        "Does RingCentral integrate with Salesforce?",
        extractor=HeuristicExtractor(),
        catalog=_catalog(),
    )
    assert extracted.source == "heuristic"
    assert extracted.kind == "factual"


def test_get_extractor_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown query extractor"):
        get_extractor("spacy")


def test_openai_extractor_uses_nano_model_by_default() -> None:
    class _Completions:
        def __init__(self) -> None:
            self.model = None

        def create(self, **kwargs):
            self.model = kwargs["model"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"kind": "factual", "providers": [], "topics": []}'
                        )
                    )
                ]
            )

    completions = _Completions()

    class _Client:
        chat = SimpleNamespace(completions=completions)

    OpenAIExtractor(client=_Client()).extract("hello", catalog=_catalog())
    assert completions.model == "gpt-4.1-nano"
