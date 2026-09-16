from app.ingestion.normalizer import normalize_post
from app.models.schemas import RawPost


def test_builds_heading_paths_and_keeps_intro() -> None:
    post = RawPost(
        id="1",
        title="Nextiva",
        slug="nextiva",
        html=(
            "<p>Overview text.</p>"
            "<h2>Pricing</h2><p>Three plans.</p>"
            "<h3>Core</h3><p>Starts at $15.</p>"
        ),
        content_type="provider",
        provider="Nextiva",
    )
    document = normalize_post(post)
    assert [section.heading for section in document.sections] == ["Nextiva", "Pricing", "Core"]
    assert document.sections[0].heading_path == ["Nextiva"]
    assert document.sections[1].heading_path == ["Nextiva", "Pricing"]
    assert document.sections[2].heading_path == ["Nextiva", "Pricing", "Core"]
    assert "Overview text." in document.sections[0].text
    assert "Starts at $15." in document.sections[2].text


def test_user_review_is_one_document_with_provider() -> None:
    post = RawPost(
        id="106948",
        title="We've had a positive experience with...",
        slug="weve-had-a-positive-experience-with",
        html="Reliable phones.\n\nHelpful support.",
        content_type="review",
        provider="Nextiva",
        url="https://getvoip.dev/?user-reviews=example",
    )
    document = normalize_post(post)
    assert document.content_type == "review"
    assert document.provider == "Nextiva"
    assert len(document.sections) == 1
    assert document.sections[0].heading_path == [post.title]
    assert "Reliable phones." in document.sections[0].text
    assert "Helpful support." in document.sections[0].text
