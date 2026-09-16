from app.ingestion.parser import parse_blocks


def test_splits_headings_lists_and_tables() -> None:
    html = """
    <h2>Pricing</h2>
    <p>Three plans.</p>
    <ul><li>Core</li><li>Engage</li></ul>
    <table>
      <tr><th>Plan</th><th>Price</th></tr>
      <tr><td>Core</td><td>$15</td></tr>
    </table>
    """
    blocks = parse_blocks(html)
    kinds = [block.kind for block in blocks]
    assert kinds == ["heading", "paragraph", "list", "table"]
    assert blocks[0].level == 2
    assert blocks[0].text == "Pricing"
    assert "- Core" in blocks[2].text
    assert "Plan | Price" in blocks[3].text
    assert "Core | $15" in blocks[3].text


def test_plain_text_review_becomes_paragraphs() -> None:
    text = "First paragraph.\n\nSecond paragraph."
    blocks = parse_blocks(text)
    assert [block.kind for block in blocks] == ["paragraph", "paragraph"]
    assert blocks[0].text == "First paragraph."
    assert blocks[1].text == "Second paragraph."
