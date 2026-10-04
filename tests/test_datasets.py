import numpy as np
import pytest

from ml_from_scratch import make_blobs, make_moons, train_test_split


def test_make_moons_shape_and_balance() -> None:
    X, y = make_moons(101, noise=0.0, random_state=0)
    assert X.shape == (101, 2)
    assert abs(int((y == 0).sum()) - int((y == 1).sum())) <= 1
    # noise-free points lie on the two unit circles
    upper = X[y == 0]
    assert np.allclose(np.linalg.norm(upper, axis=1), 1.0)


def test_make_blobs_and_split() -> None:
    X, y = make_blobs(90, 3, random_state=0)
    assert X.shape == (90, 2)
    assert np.bincount(y).tolist() == [30, 30, 30]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    assert len(Xte) == 18
    assert len(Xtr) + len(Xte) == 90
    assert len(ytr) == len(Xtr)
    assert len(yte) == len(Xte)
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5)


def test_reproducible() -> None:
    a, _ = make_moons(50, random_state=7)
    b, _ = make_moons(50, random_state=7)
    assert np.array_equal(a, b)
