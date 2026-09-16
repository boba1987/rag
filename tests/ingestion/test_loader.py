from app.ingestion.loader import get_raw_post, load_all_raw_posts


def test_loads_all_three_fixture_types() -> None:
    posts = load_all_raw_posts()
    types = {post.content_type for post in posts}
    assert types == {"article", "review", "provider"}
    assert len(posts) == 32


def test_get_raw_post_maps_provider_and_review() -> None:
    provider = get_raw_post(8019, "provider")
    assert provider.title == "Nextiva"
    assert provider.provider == "Nextiva"
    assert "<h2" in provider.html

    review = get_raw_post(106948, "review")
    assert review.provider == "Nextiva"
    assert "Nextiva" in review.html
