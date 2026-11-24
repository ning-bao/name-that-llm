import numpy as np

from nllm import centring


def test_centre_uses_the_other_rows_and_leaves_singletons():
    scores = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0], [7.0, 1.0]])
    clusters = np.array([0, 0, 0, 1])
    out = centring.centre(scores, clusters, 0.5, cavity=True)
    assert np.allclose(out[0], scores[0] - 0.5 * scores[[1, 2]].mean(axis=0))
    assert np.allclose(out[3], scores[3])  # a singleton has no cluster mean
    plain = centring.centre(scores, clusters, 0.5, cavity=False)
    assert np.allclose(plain[1], scores[1] - 0.5 * scores[:3].mean(axis=0))
    assert np.array_equal(centring.centre(scores, clusters, 0.0), scores)


def test_content_clusters_group_by_topic():
    texts = ["The volcano erupted and lava flowed down the volcano slopes.",
             "Lava from the volcano covered the slopes after it erupted.",
             "Bread dough rises when yeast ferments sugar in the dough.",
             "Yeast ferments the sugar, so the bread dough rises."]
    clusters = centring.content_clusters(texts, 0.85)
    assert clusters[0] == clusters[1] and clusters[2] == clusters[3] and clusters[0] != clusters[2]
    assert len(centring.content_clusters(texts[:1], 0.85)) == 1
