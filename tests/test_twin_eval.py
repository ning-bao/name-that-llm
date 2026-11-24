import importlib.util

import numpy as np

from nllm.paths import ROOT


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_twin_scores_centre_the_saved_outputs_within_groups_of_two_or_more():
    script = load_script("twin_eval")
    rng = np.random.default_rng(0)
    width, labels = 5, 4
    weight, bias = rng.normal(size=(labels, width)), rng.normal(size=labels)
    feats = rng.normal(size=(5, width))
    plain = rng.normal(size=(5, labels))
    groups = np.array(["a", "b", "a", "c", "a"])  # one group of three and two single rows
    saved = feats @ weight.T + bias  # what a fit saves: the centred head on uncentred features
    got = script.twin_scores(plain, saved, np.tile(bias, (5, 1)), groups, twin=0.35)

    expected = script.log_softmax(plain)
    in_a = groups == "a"
    centred = (feats[in_a] - feats[in_a].mean(axis=0)) @ weight.T + bias  # the head on centred features
    expected[in_a] += 0.35 * script.log_softmax(centred)
    assert np.allclose(got, expected)
    assert np.allclose(got[~in_a], script.log_softmax(plain)[~in_a])  # single rows keep the plain score
