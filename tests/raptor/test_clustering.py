from app.raptor.clustering import choose_cluster_count, cluster_embeddings, cosine_similarity


def test_cosine_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == 1.0


def test_cosine_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_choose_cluster_count_scales_with_leaves() -> None:
    assert choose_cluster_count(0) == 0
    assert choose_cluster_count(1) == 1
    assert choose_cluster_count(4) == 2
    assert choose_cluster_count(8) == 2
    assert choose_cluster_count(9) == 3


def test_cluster_embeddings_empty_and_single() -> None:
    assert cluster_embeddings([]) == []
    assert cluster_embeddings([[1.0, 0.0]]) == [[0]]


def test_cluster_embeddings_separates_two_directions() -> None:
    vectors = [
        [1.0, 0.0],
        [0.95, 0.05],
        [1.05, -0.02],
        [0.0, 1.0],
        [0.04, 0.96],
        [-0.03, 1.02],
    ]
    groups = cluster_embeddings(vectors, k=2)
    assert len(groups) == 2
    grouped = {frozenset(group) for group in groups}
    assert grouped == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}


def test_cluster_embeddings_k_equals_n_are_singletons() -> None:
    groups = cluster_embeddings([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], k=3)
    assert groups == [[0], [1], [2]]
