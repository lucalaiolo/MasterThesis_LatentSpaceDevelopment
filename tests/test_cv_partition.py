import numpy as np
import pytest

from vae_analysis.hmm_pipeline import kfold_subject_splits


@pytest.mark.parametrize("n_videos,n_splits", [(38, 5), (38, 38), (10, 3), (7, 7)])
def test_folds_partition(n_videos, n_splits):
    folds = kfold_subject_splits(n_videos, n_splits, seed=0)
    assert len(folds) == n_splits
    counts = np.zeros(n_videos, dtype=int)
    for fold in folds:
        counts[np.asarray(fold, dtype=int)] += 1
    assert (counts == 1).all()                       # each subject validated once
    sizes = sorted(len(f) for f in folds)
    assert max(sizes) - min(sizes) <= 1              # balanced to within one


def test_deterministic():
    a = [f.tolist() for f in kfold_subject_splits(38, 5, seed=0)]
    b = [f.tolist() for f in kfold_subject_splits(38, 5, seed=0)]
    assert a == b


def test_rejects_too_many_splits():
    with pytest.raises(ValueError):
        kfold_subject_splits(5, 10, seed=0)
