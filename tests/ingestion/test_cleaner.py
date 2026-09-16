from app.ingestion.cleaner import clean_html


def test_strips_scripts_images_and_shortcodes() -> None:
    html = (
        "<p>Hello [cta id=1]world</p>"
        "<script>alert(1)</script>"
        "<img src='x.jpg' alt='ad'>"
        "<p class='advertisement'>Buy now</p>"
    )
    cleaned = clean_html(html)
    assert "Hello" in cleaned
    assert "world" in cleaned
    assert "[cta" not in cleaned
    assert "script" not in cleaned.lower()
    assert "img" not in cleaned.lower()
    assert "Buy now" not in cleaned


def test_removes_jump_to_anchor_list() -> None:
    html = (
        "<p>Intro</p>"
        "<strong>Jump to</strong>"
        "<ul><li><a href='#Pricing'>Pricing</a></li></ul>"
        "<h2 id='Pricing'>Pricing</h2>"
        "<p>Plans start at $15.</p>"
    )
    cleaned = clean_html(html)
    assert "Jump to" not in cleaned
    assert 'href="#Pricing"' not in cleaned
    assert "Pricing" in cleaned
    assert "Plans start at $15." in cleaned
