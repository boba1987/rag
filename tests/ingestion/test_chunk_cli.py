from app.ingestion.chunk import main
from app.models.schemas import NormalizedDocument, Section


def _write_normalized(directory) -> None:
    document = NormalizedDocument(
        id="8019",
        title="Nextiva",
        content_type="provider",
        provider="Nextiva",
        sections=[
            Section(heading="Pricing", heading_path=["Nextiva", "Pricing"], text="Three plans start at $15."),
            Section(heading="Support", heading_path=["Nextiva", "Support"], text="24/7 phone and chat."),
        ],
    )
    path = directory / "provider-8019-nextiva.json"
    path.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")


def test_cli_writes_fixed_size_to_its_directory(tmp_path, monkeypatch) -> None:
    normalized = tmp_path / "normalized"
    chunked = tmp_path / "chunked_fixed"
    normalized.mkdir()
    _write_normalized(normalized)
    monkeypatch.setattr("app.ingestion.chunk.NORMALIZED_DIR", normalized)
    monkeypatch.setattr("app.ingestion.chunk.chunked_dir", lambda name=None: chunked)

    assert main(["--all", "--chunker", "fixed_size"]) == 0
    written = chunked / "provider-8019-nextiva.json"
    assert written.exists()
    assert "fixed_01" in written.read_text(encoding="utf-8")
    assert not (chunked / "parents").exists()


def test_cli_writes_parent_child_children_and_parents(tmp_path, monkeypatch) -> None:
    normalized = tmp_path / "normalized"
    chunked = tmp_path / "chunked_parent_child"
    normalized.mkdir()
    _write_normalized(normalized)
    monkeypatch.setattr("app.ingestion.chunk.NORMALIZED_DIR", normalized)
    monkeypatch.setattr("app.ingestion.chunk.chunked_dir", lambda name=None: chunked)

    assert main(["--id", "8019", "--chunker", "parent_child"]) == 0
    children = (chunked / "provider-8019-nextiva.json").read_text(encoding="utf-8")
    parents = (chunked / "parents" / "provider-8019-nextiva.json").read_text(encoding="utf-8")
    assert "child_01" in children
    assert "pricing_parent" in parents
    assert "parent_id" in children
