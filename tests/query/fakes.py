from app.query.extractor import QueryExtraction, QueryExtractor


class ScriptedExtractor(QueryExtractor):
    name = "openai"

    def __init__(self, by_query: dict[str, QueryExtraction] | None = None) -> None:
        self._by_query = by_query or {}

    def extract(self, query: str, catalog=None) -> QueryExtraction:
        if query in self._by_query:
            return self._by_query[query]
        for fragment, extraction in self._by_query.items():
            if fragment in query:
                return extraction
        return QueryExtraction(kind="factual", source="openai")


PRICING = QueryExtraction(kind="pricing", providers=["RingCentral"], topics=["pricing"], source="openai")
REVIEW = QueryExtraction(kind="review", providers=["Nextiva"], topics=["support"], source="openai")
COMPARISON = QueryExtraction(
    kind="comparison",
    providers=["Five9", "Dialpad"],
    topics=[],
    source="openai",
)
MULTI_HOP = QueryExtraction(
    kind="multi-hop",
    providers=["RingCentral", "Nextiva"],
    topics=["pricing", "Salesforce integration"],
    source="openai",
)
FACTUAL_SF = QueryExtraction(
    kind="factual",
    providers=["RingCentral"],
    topics=["Integration"],
    source="openai",
)

DEFAULT_SCRIPTED = ScriptedExtractor(
    {
        "How much does RingCentral cost?": PRICING,
        "how much Ring Central costs": PRICING,
        "What do customers say about Nextiva support?": REVIEW,
        "Who is Five9 better for compared with Dialpad?": COMPARISON,
        "Compare RingCentral and Nextiva pricing and Salesforce integrations.": MULTI_HOP,
        "Does RingCentral integrate with Salesforce?": FACTUAL_SF,
    }
)
