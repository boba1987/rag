from app.ingestion.loader import get_raw_post, load_all_raw_posts, load_raw_posts_from_file


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


def test_load_raw_posts_from_file_uses_provider_shape(tmp_path) -> None:
    path = tmp_path / "providers.json"
    path.write_text(
        '[{"id": 1, "name": "Acme", "slug": "acme", "content": "<p>Hello</p>"}]\n',
        encoding="utf-8",
    )
    posts = load_raw_posts_from_file(path, "provider")
    assert len(posts) == 1
    assert posts[0].id == "1"
    assert posts[0].content_type == "provider"
    assert posts[0].provider == "Acme"
    assert posts[0].html == "<p>Hello</p>"


def test_load_raw_posts_from_file_missing_path(tmp_path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        load_raw_posts_from_file(tmp_path / "missing.json", "article")
