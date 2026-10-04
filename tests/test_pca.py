import numpy as np
import pytest

from ml_from_scratch import PCA


@pytest.fixture
def X() -> np.ndarray:
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(300, 2)) * np.array([5.0, 1.5])
    mix = rng.normal(size=(2, 5))
    return latent @ mix + rng.normal(0, 0.05, size=(300, 5)) + 10.0


def test_diagonal_covariance_known_result() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5000, 3)) * np.array([3.0, 2.0, 0.5])
    pca = PCA().fit(X)
    assert pca.explained_variance_ == pytest.approx([9.0, 4.0, 0.25], rel=0.1)
    assert np.allclose(np.abs(pca.components_), np.eye(3), atol=0.05)


def test_eigendecomposition_of_covariance(X: np.ndarray) -> None:
    pca = PCA().fit(X)
    vals, vecs = np.linalg.eigh(np.cov(X, rowvar=False))
    vals, vecs = vals[::-1], vecs[:, ::-1]
    assert np.allclose(pca.explained_variance_, vals)
    assert np.allclose(np.abs(pca.components_), np.abs(vecs.T), atol=1e-8)


def test_properties(X: np.ndarray) -> None:
    pca = PCA(3).fit(X)
    assert pca.components_.shape == (3, 5)
    assert np.allclose(pca.components_ @ pca.components_.T, np.eye(3))  # orthonormal
    assert np.all(np.diff(pca.explained_variance_) <= 0)
    assert PCA().fit(X).explained_variance_ratio_.sum() == pytest.approx(1.0)
    Z = pca.transform(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Z.var(axis=0, ddof=1), pca.explained_variance_)
    assert np.allclose(np.cov(Z, rowvar=False), np.diag(pca.explained_variance_), atol=1e-9)


def test_reconstruction(X: np.ndarray) -> None:
    full = PCA().fit(X)
    assert np.allclose(full.inverse_transform(full.transform(X)), X)
    two = PCA(2).fit(X)
    err = np.linalg.norm(X - two.inverse_transform(two.transform(X))) ** 2
    # Eckart-Young: residual = sum of discarded squared singular values
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    assert err == pytest.approx((s[2:] ** 2).sum())


def test_deterministic_sign(X: np.ndarray) -> None:
    pca = PCA(2).fit(X)
    pivot = np.abs(pca.components_).argmax(axis=1)
    assert np.all(pca.components_[np.arange(2), pivot] > 0)


def test_matches_sklearn(X: np.ndarray) -> None:
    decomposition = pytest.importorskip("sklearn.decomposition")
    ours = PCA(2).fit(X)
    ref = decomposition.PCA(2).fit(X)
    assert np.allclose(np.abs(ours.components_), np.abs(ref.components_))
    assert np.allclose(ours.explained_variance_, ref.explained_variance_)
    assert np.allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_)
    assert np.allclose(np.abs(ours.transform(X)), np.abs(ref.transform(X)))


def test_validation() -> None:
    with pytest.raises(ValueError):
        PCA(0)
    with pytest.raises(ValueError):
        PCA(4).fit(np.zeros((10, 3)))
    with pytest.raises(ValueError):
        PCA().fit(np.zeros((1, 3)))
    with pytest.raises(RuntimeError):
        PCA().transform(np.zeros((2, 2)))
