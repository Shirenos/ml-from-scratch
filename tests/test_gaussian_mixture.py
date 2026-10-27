import numpy as np
import pytest

from ml_from_scratch import GaussianMixture, make_blobs


def _overlapping(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.multivariate_normal([0.0, 0.0], [[2.0, 1.2], [1.2, 1.0]], size=150)
    b = rng.multivariate_normal([2.0, 1.0], [[0.5, -0.2], [-0.2, 0.8]], size=100)
    c = rng.multivariate_normal([-1.0, 3.0], [[1.0, 0.0], [0.0, 0.3]], size=80)
    return np.vstack([a, b, c])


def _manual_log_density(X: np.ndarray, model: GaussianMixture) -> np.ndarray:
    d = X.shape[1]
    dens = np.zeros(len(X))
    for w, mu, cov in zip(model.weights_, model.means_, model.covariances_, strict=True):
        diff = X - mu
        maha = np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov), diff)
        norm = np.sqrt((2.0 * np.pi) ** d * np.linalg.det(cov))
        dens += w * np.exp(-0.5 * maha) / norm
    return np.log(dens)


@pytest.mark.parametrize("init", ["kmeans", "random"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_log_likelihood_never_decreases(init: str, seed: int) -> None:
    X = _overlapping(seed)
    model = GaussianMixture(3, init=init, reg_covar=0.0, tol=0.0, max_iter=60, random_state=seed)  # type: ignore[arg-type]
    hist = np.array(model.fit(X).log_likelihood_history_)
    assert len(hist) >= 2
    assert np.all(np.diff(hist) >= -1e-10)
    assert model.lower_bound_ == pytest.approx(model.score(X))


def test_single_component_is_the_gaussian_mle() -> None:
    X = _overlapping()
    n, d = X.shape
    model = GaussianMixture(1, reg_covar=0.0).fit(X)
    mean = X.mean(axis=0)
    cov = (X - mean).T @ (X - mean) / n  # biased: the MLE divides by n
    assert np.allclose(model.weights_, [1.0])
    assert np.allclose(model.means_[0], mean)
    assert np.allclose(model.covariances_[0], cov)
    # maximised log-likelihood has the closed form -(n/2) (d log 2 pi + log det S + d)
    expected = -0.5 * (d * np.log(2.0 * np.pi) + np.log(np.linalg.det(cov)) + d)
    assert model.score(X) == pytest.approx(expected)
    assert model.converged_


def test_score_samples_matches_the_density_formula() -> None:
    X = _overlapping()
    model = GaussianMixture(3, random_state=0).fit(X)
    assert np.allclose(model.score_samples(X), _manual_log_density(X, model))


def test_fit_is_a_fixed_point_of_the_m_step() -> None:
    X = _overlapping()
    model = GaussianMixture(3, reg_covar=0.0, tol=1e-12, max_iter=1000, random_state=0).fit(X)
    resp = model.predict_proba(X)
    nk = resp.sum(axis=0)
    assert np.allclose(model.weights_, nk / len(X), atol=1e-6)
    assert np.allclose(model.means_, resp.T @ X / nk[:, None], atol=1e-5)


def test_recovers_separated_components() -> None:
    true = np.array([[-6.0, -6.0], [0.0, 6.0], [6.0, -6.0]])
    X, y = make_blobs(600, true, cluster_std=0.7, random_state=0)
    model = GaussianMixture(3, random_state=0).fit(X)
    for c in true:
        assert np.min(np.linalg.norm(model.means_ - c, axis=1)) < 0.2
    assert np.allclose(model.weights_, 1.0 / 3.0, atol=0.05)
    for cov in model.covariances_:
        assert np.allclose(cov, 0.49 * np.eye(2), atol=0.12)
    pairs = {(a, b) for a, b in zip(y, model.predict(X), strict=True)}
    assert len(pairs) == 3


def test_responsibilities_are_stable_far_from_the_data() -> None:
    X = _overlapping()
    model = GaussianMixture(3, random_state=0).fit(X)
    far = np.array([[1e4, -1e4], [0.0, 0.0]])
    proba = model.predict_proba(far)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(np.isfinite(model.score_samples(far)))


def test_more_restarts_never_hurt() -> None:
    X = _overlapping(3)
    one = GaussianMixture(3, init="random", n_init=1, random_state=5).fit(X)
    many = GaussianMixture(3, init="random", n_init=5, random_state=5).fit(X)
    assert many.lower_bound_ >= one.lower_bound_ - 1e-12


def test_information_criteria() -> None:
    X = _overlapping()
    n = len(X)
    model = GaussianMixture(3, random_state=0).fit(X)
    p = 2 + 3 * 2 + 3 * 3  # weights, means, covariances for K=3, d=2
    log_l = model.score(X) * n
    assert model.bic(X) == pytest.approx(-2.0 * log_l + p * np.log(n))
    assert model.aic(X) == pytest.approx(-2.0 * log_l + 2.0 * p)
    one = GaussianMixture(1).fit(X)
    assert model.bic(X) < one.bic(X)


def test_sample_reproduces_the_fitted_moments() -> None:
    X = _overlapping()
    model = GaussianMixture(1).fit(X)
    S, labels = model.sample(20000, random_state=0)
    assert S.shape == (20000, 2)
    assert np.all(labels == 0)
    assert np.allclose(S.mean(axis=0), model.means_[0], atol=0.05)
    assert np.allclose(np.cov(S.T, bias=True), model.covariances_[0], atol=0.08)


def test_matches_sklearn() -> None:
    sk = pytest.importorskip("sklearn.mixture")
    X = _overlapping()
    # scikit-learn's default tol=1e-3 stops early, so run both to convergence
    ours = GaussianMixture(3, n_init=5, tol=1e-10, max_iter=2000, random_state=0).fit(X)
    ref = sk.GaussianMixture(3, n_init=5, tol=1e-10, max_iter=2000, random_state=0).fit(X)
    assert ours.score(X) == pytest.approx(ref.score(X), abs=1e-6)
    order = np.argsort(ours.means_[:, 0])
    ref_order = np.argsort(ref.means_[:, 0])
    assert np.allclose(ours.means_[order], ref.means_[ref_order], atol=1e-4)


def test_validation() -> None:
    with pytest.raises(ValueError):
        GaussianMixture(0)
    with pytest.raises(ValueError):
        GaussianMixture(2, reg_covar=-1.0)
    with pytest.raises(ValueError):
        GaussianMixture(2, init="nope")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        GaussianMixture(2).predict(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        GaussianMixture(5).fit(np.zeros((3, 2)))
    model = GaussianMixture(2, random_state=0).fit(_overlapping())
    with pytest.raises(ValueError):
        model.predict(np.zeros((2, 3)))
