import numpy as np
import pytest

from ml_from_scratch import KMeans, make_blobs


def test_tiny_known_example() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])
    model = KMeans(2, random_state=0).fit(X)
    centers = model.cluster_centers_[np.argsort(model.cluster_centers_[:, 0])]
    assert np.allclose(centers, [[0.0, 0.5], [10.0, 10.5]])
    assert model.inertia_ == pytest.approx(4 * 0.25)
    assert model.labels_[0] == model.labels_[1] != model.labels_[2] == model.labels_[3]


def test_recovers_well_separated_blobs() -> None:
    true = np.array([[-6.0, -6.0], [0.0, 6.0], [6.0, -6.0]])
    X, y = make_blobs(300, true, cluster_std=0.5, random_state=0)
    model = KMeans(3, random_state=0).fit(X)
    for c in true:
        assert np.min(np.linalg.norm(model.cluster_centers_ - c, axis=1)) < 0.3
    # labels are a relabelling of the truth
    pairs = {(a, b) for a, b in zip(y, model.labels_, strict=True)}
    assert len(pairs) == 3


def test_inertia_is_consistent_with_labels_and_centers() -> None:
    X, _ = make_blobs(200, 4, random_state=1)
    model = KMeans(4, random_state=1).fit(X)
    manual = ((X - model.cluster_centers_[model.labels_]) ** 2).sum()
    assert model.inertia_ == pytest.approx(manual)
    # centres are the means of their clusters (fixed point of Lloyd)
    for j in range(4):
        assert np.allclose(model.cluster_centers_[j], X[model.labels_ == j].mean(axis=0))


def test_predict_matches_labels() -> None:
    X, _ = make_blobs(150, 3, random_state=2)
    model = KMeans(3, random_state=0).fit(X)
    assert np.array_equal(model.predict(X), model.labels_)
    assert np.array_equal(KMeans(3, random_state=0).fit_predict(X), model.labels_)


def test_more_restarts_never_hurt() -> None:
    X, _ = make_blobs(300, 6, cluster_std=1.5, random_state=3)
    one = KMeans(6, n_init=1, init="random", random_state=0).fit(X)
    many = KMeans(6, n_init=20, init="random", random_state=0).fit(X)
    assert many.inertia_ <= one.inertia_ + 1e-9


def test_duplicate_points_and_k_equals_n() -> None:
    X = np.ones((5, 2))
    assert KMeans(2, random_state=0).fit(X).inertia_ == 0.0
    Y = np.arange(6, dtype=float).reshape(3, 2)
    assert KMeans(3, random_state=0).fit(Y).inertia_ == pytest.approx(0.0)


def test_matches_sklearn_inertia() -> None:
    cluster = pytest.importorskip("sklearn.cluster")
    centers = [[-8, -8], [-8, 8], [0, 0], [8, -8], [8, 8]]
    X, _ = make_blobs(400, centers, cluster_std=0.8, random_state=4)
    ours = KMeans(5, n_init=10, random_state=0).fit(X)
    ref = cluster.KMeans(5, n_init=10, random_state=0).fit(X)
    assert ours.inertia_ == pytest.approx(ref.inertia_, rel=1e-6)


def test_validation() -> None:
    with pytest.raises(ValueError):
        KMeans(5).fit(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        KMeans(0)
    with pytest.raises(RuntimeError):
        KMeans().predict(np.zeros((2, 2)))
