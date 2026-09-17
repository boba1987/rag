from __future__ import annotations

import math
import random
from collections import defaultdict


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity of two vectors. Zero vectors score 0.0."""
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def choose_cluster_count(n: int) -> int:
    """Pick k so clusters stay small enough to summarize. One cluster when n < 2."""
    if n <= 1:
        return n
    return max(2, min(n, math.ceil(n / 4)))


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


def _mean(vectors: list[list[float]]) -> list[float]:
    width = len(vectors[0])
    totals = [0.0] * width
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    count = float(len(vectors))
    return [total / count for total in totals]


def _farthest_index(unit: list[list[float]], centroids: list[list[float]]) -> int:
    best_index = 0
    best_distance = -1.0
    for index, vector in enumerate(unit):
        nearest = max(cosine_similarity(vector, centroid) for centroid in centroids)
        distance = 1.0 - nearest
        if distance > best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _init_centroids(unit: list[list[float]], k: int, rng: random.Random) -> list[list[float]]:
    first = rng.randrange(len(unit))
    centroids = [unit[first]]
    while len(centroids) < k:
        weights = []
        for vector in unit:
            nearest = max(cosine_similarity(vector, centroid) for centroid in centroids)
            weights.append((1.0 - nearest) ** 2)
        total = sum(weights)
        if total == 0.0:
            centroids.append(unit[_farthest_index(unit, centroids)])
            continue
        pick = rng.random() * total
        running = 0.0
        chosen = len(unit) - 1
        for index, weight in enumerate(weights):
            running += weight
            if running >= pick:
                chosen = index
                break
        centroids.append(unit[chosen])
    return centroids


def cluster_embeddings(
    vectors: list[list[float]],
    k: int | None = None,
    *,
    max_rounds: int = 25,
    seed: int = 0,
) -> list[list[int]]:
    """Group vector indices with cosine k-means. Empty input returns no groups."""
    count = len(vectors)
    if count == 0:
        return []
    if count == 1:
        return [[0]]

    clusters = choose_cluster_count(count) if k is None else k
    clusters = max(1, min(clusters, count))
    if clusters == 1:
        return [list(range(count))]
    if clusters == count:
        return [[index] for index in range(count)]

    unit = [_normalize(vector) for vector in vectors]
    rng = random.Random(seed)
    centroids = _init_centroids(unit, clusters, rng)
    assignment = [0] * count

    for _ in range(max_rounds):
        changed = False
        groups: dict[int, list[int]] = defaultdict(list)
        for index, vector in enumerate(unit):
            nearest = max(
                range(clusters),
                key=lambda centroid_index: cosine_similarity(vector, centroids[centroid_index]),
            )
            if assignment[index] != nearest:
                changed = True
                assignment[index] = nearest
            groups[nearest].append(index)

        for centroid_index in range(clusters):
            members = groups.get(centroid_index, [])
            if members:
                centroids[centroid_index] = _normalize(_mean([unit[index] for index in members]))
            else:
                centroids[centroid_index] = unit[_farthest_index(unit, centroids)]
        if not changed:
            break

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, cluster_index in enumerate(assignment):
        grouped[cluster_index].append(index)
    return [members for _, members in sorted(grouped.items()) if members]
