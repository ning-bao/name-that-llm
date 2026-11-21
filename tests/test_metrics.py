import numpy as np
import pytest

from nllm.metrics import balanced_accuracy, interval, per_class_recall, prompt_bootstrap


def test_balanced_accuracy_is_mean_class_recall():
    y_true = ["a", "a", "a", "a", "b"]
    y_pred = ["a", "a", "a", "a", "a"]
    assert balanced_accuracy(y_true, y_pred, labels=["a", "b"]) == pytest.approx(0.5)
    assert per_class_recall(y_true, y_pred, labels=["a", "b"]).to_dict() == {"a": 1.0, "b": 0.0}


def test_absent_classes_are_ignored():
    assert balanced_accuracy(["a", "a"], ["a", "b"], labels=["a", "b", "c"]) == pytest.approx(0.5)


def test_bootstrap_paired_difference_of_identical_predictions_is_zero():
    rng = np.random.default_rng(0)
    groups = np.repeat([f"p{i}" for i in range(20)], 3)
    y_true = np.tile(["a", "b", "c"], 20)
    y_pred = rng.choice(["a", "b", "c"], size=60)
    diff = prompt_bootstrap(y_true, y_pred, groups, y_pred_baseline=y_pred, n_draws=200)
    assert diff.shape == (200,)
    assert np.allclose(diff, 0)


def test_bootstrap_centres_on_point_estimate():
    groups = np.repeat([f"p{i}" for i in range(40)], 2)
    y_true = np.tile(["a", "b"], 40)
    y_pred = np.where(np.arange(80) % 4 == 0, "b", y_true)
    draws = prompt_bootstrap(y_true, y_pred, groups, n_draws=2000)
    lo, hi = interval(draws)
    assert lo < balanced_accuracy(y_true, y_pred, labels=["a", "b"]) < hi
