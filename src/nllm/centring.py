"""Test-time soft centring inside content clusters (plan E6, 2026-11-08).

Passages that answer the same prompt share a topic offset in every model's scores. Grouping test passages by
content (TF-IDF on topic words, never on the classifier's features) and subtracting a share of each group's mean
score removes part of that offset. The mean is a cavity estimate: for each passage it averages only the *other*
members of its cluster, because with two or three answers per prompt the plain mean is mostly the passage's own
author signal (forum threads 750213 and 752708). Nothing here reads labels, and on the test it reads only the test
passages themselves. Validation: scripts/test_time_centring.py, with prompts cut to a few answers each, since the
hidden test's answers per prompt are unknown and likely few.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer


def content_clusters(texts, cut: float) -> np.ndarray:
    """Cluster ids from average-linkage clustering of TF-IDF vectors (English stop words removed, sublinear tf,
    fitted on `texts` alone) at cosine distance `cut`. Topic grouping only: lowercasing is fine here, since these
    vectors never reach a classifier."""
    texts = list(texts)
    if len(texts) < 2:
        return np.zeros(len(texts), dtype=int)
    x = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)  # rows L2-normalised
    distance = np.clip(1.0 - (x @ x.T).toarray(), 0.0, None)
    np.fill_diagonal(distance, 0.0)
    return AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                   distance_threshold=cut).fit_predict(distance)


def centre(scores: np.ndarray, clusters: np.ndarray, weight: float, cavity: bool = True) -> np.ndarray:
    """scores minus `weight` x the mean score of each row's cluster: of the other rows (cavity) or of all rows.
    Singletons are unchanged."""
    scores = np.asarray(scores, dtype=np.float64)
    if weight == 0:
        return scores.copy()
    _, idx, sizes = np.unique(clusters, return_inverse=True, return_counts=True)
    sums = np.zeros((len(sizes), scores.shape[1]))
    np.add.at(sums, idx, scores)
    n = sizes[idx][:, None].astype(float)
    mean = (sums[idx] - scores) / np.maximum(n - 1, 1) if cavity else sums[idx] / n
    return np.where(n >= 2, scores - weight * mean, scores)
