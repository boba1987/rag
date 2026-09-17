import json

from app.ingestion.index import _load_chunk_files
from app.models.schemas import Chunk


def _chunk_row() -> dict:
    return Chunk(
        id="provider_8019_pricing_child_01",
        document_id="8019",
        content_type="provider",
        provider="Nextiva",
        title="Nextiva",
        section="Pricing",
        heading_path=["Nextiva", "Pricing"],
        text="Core starts at $15.",
        parent_id="provider_8019_pricing_parent",
    ).model_dump()


def test_load_chunk_files_skips_parents_subdirectory(tmp_path) -> None:
    (tmp_path / "provider-8019-nextiva.json").write_text(
        json.dumps([_chunk_row()]) + "\n", encoding="utf-8"
    )
    parents = tmp_path / "parents"
    parents.mkdir()
    (parents / "provider-8019-nextiva.json").write_text(
        json.dumps([{**_chunk_row(), "id": "provider_8019_pricing_parent", "parent_id": None}])
        + "\n",
        encoding="utf-8",
    )
    files = _load_chunk_files(None, None, tmp_path)
    assert [path.name for path, _chunks in files] == ["provider-8019-nextiva.json"]
    assert files[0][1][0].id.endswith("child_01")
