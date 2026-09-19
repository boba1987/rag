from types import SimpleNamespace
from pathlib import Path

from app.query.catalog import catalog_from_qdrant, load_catalog


class _FakeQdrant:
    def __init__(self, points: list[dict], exists: bool = True) -> None:
        self._points = [SimpleNamespace(payload=payload) for payload in points]
        self._exists = exists

    def collection_exists(self, name: str) -> bool:
        return self._exists

    def scroll(self, **kwargs):
        return self._points, None


def test_load_catalog_from_normalized_uses_payload_provider_only(tmp_path: Path) -> None:
    (tmp_path / "provider-1.json").write_text(
        """
        {
          "id": "1",
          "title": "RingCentral",
          "content_type": "provider",
          "provider": "RingCentral",
          "sections": [
            {"heading": "Plans and Pricing", "heading_path": ["RingCentral", "Plans and Pricing"], "text": "Salesforce integration available."}
          ]
        }
        """
    )
    (tmp_path / "article-2.json").write_text(
        """
        {
          "id": "2",
          "title": "Five9 vs Dialpad: How the Two Platforms Stack Up",
          "content_type": "article",
          "provider": null,
          "sections": [
            {"heading": "Overview", "heading_path": ["Overview"], "text": "Feature comparison."}
          ]
        }
        """
    )
    catalog = load_catalog(normalized_dir=tmp_path)
    assert catalog.providers == ("RingCentral",)
    assert "Plans and Pricing" in catalog.headings
    assert catalog.match_providers("ring central vs five9") == ["RingCentral"]
    assert catalog.allows_topic("pricing")
    assert catalog.allows_topic("Salesforce integration")


def test_technology_vs_title_is_not_a_provider(tmp_path: Path) -> None:
    (tmp_path / "article.json").write_text(
        """
        {
          "id": "3",
          "title": "VoIP vs Landline: What’s the Difference",
          "content_type": "article",
          "provider": null,
          "sections": [
            {"heading": "What is VoIP?", "heading_path": ["What is VoIP?"], "text": "Voice over Internet Protocol."}
          ]
        }
        """
    )
    catalog = load_catalog(normalized_dir=tmp_path)
    assert catalog.providers == ()
    assert catalog.match_providers("what is voip?") == []


def test_catalog_from_qdrant_uses_payload_provider_only() -> None:
    catalog = catalog_from_qdrant(
        client=_FakeQdrant(
            [
                {
                    "provider": "Nextiva",
                    "title": "Nextiva",
                    "section": "Pricing",
                    "text": "Nextiva plans and pricing.",
                },
                {
                    "provider": None,
                    "title": "VoIP vs Landline: What’s the Difference",
                    "section": "What is VoIP?",
                    "text": "Voice over Internet Protocol.",
                },
            ]
        )
    )
    assert catalog.providers == ("Nextiva",)
    assert catalog.match_providers("How much does Nextiva cost?") == ["Nextiva"]
    assert catalog.match_providers("what is voip?") == []
    assert "What is VoIP?" in catalog.headings


def test_unknown_name_is_not_a_provider() -> None:
    catalog = catalog_from_qdrant(
        client=_FakeQdrant(
            [{"provider": "RingCentral", "title": "RingCentral", "section": "Overview", "text": ""}]
        )
    )
    assert catalog.match_providers("How much does AcmePBX cost?") == []


def test_missing_collection_is_empty() -> None:
    catalog = catalog_from_qdrant(client=_FakeQdrant([], exists=False))
    assert catalog.providers == ()
