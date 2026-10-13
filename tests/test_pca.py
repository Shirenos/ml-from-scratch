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
    for row in pca.components_:
        nz = np.flatnonzero(row)
        assert nz.size > 0
        assert row[nz[0]] > 0


def test_sign_flip_zero_pivot_does_not_zero_component() -> None:
    """A zero pivot must not wipe the component via ``np.sign(0)``.

    Using the first non-zero loading as the sign reference keeps the vector
    intact when the leading entry is exactly 0 (the classic failure mode of
    ``row *= np.sign(row[0])``).
    """
    from ml_from_scratch.pca import _flip_singular_vectors

    # Old buggy convention: multiply by sign of entry 0 -> zeros the rows.
    vt = np.array([[0.0, -0.6, -0.8], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    buggy = vt * np.sign(vt[:, 0])[:, None]
    assert np.allclose(buggy, 0.0)

    flipped = _flip_singular_vectors(vt)
    assert flipped[0, 0] == 0.0
    assert flipped[0, 1] > 0  # first non-zero became positive
    assert np.allclose(flipped[0], -vt[0])
    assert flipped[1, 2] > 0
    assert np.allclose(flipped[1], -vt[1])
    assert np.allclose(flipped[2], 0.0)  # all-zero row stays zero

    # Constant feature -> many components have a leading structural 0.
    rng = np.random.default_rng(0)
    X = np.column_stack([np.full(80, 3.0), rng.normal(size=(80, 2))])
    pca = PCA(2).fit(X)
    assert np.allclose(pca.components_[:, 0], 0.0, atol=1e-10)
    assert np.all(np.linalg.norm(pca.components_, axis=1) > 0.9)
    for row in pca.components_:
        nz = np.flatnonzero(np.abs(row) > 1e-12)
        assert nz.size > 0
        assert row[nz[0]] > 0


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
