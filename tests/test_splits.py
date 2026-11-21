import pytest

from nllm import splits
from nllm.paths import data_dir

needs_data = pytest.mark.skipif(
    not (data_dir() / "train.csv").exists(), reason="competition data not present"
)


def test_confirmation_set_is_locked():
    with pytest.raises(PermissionError):
        splits.confirmation_prompts()
    assert len(splits.confirmation_prompts(unlock=True)) == 30


def test_folds_are_prompt_disjoint():
    for partition in splits.partitions():
        for fold in splits.outer_folds(partition):
            assert not fold.train & fold.validation
            assert len(fold.train) == 96 and len(fold.validation) == 24
            for inner_train, inner_val in fold.inner():
                assert not inner_train & inner_val
                assert inner_train | inner_val == fold.train


@needs_data
def test_plan_matches_train_csv():
    from nllm.data import load_train

    splits.check_plan(load_train())
