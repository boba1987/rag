from app.query.inspect import inspect_preprocess, write_preprocess_inspect
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
