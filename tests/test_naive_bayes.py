import numpy as np
import pytest

from ml_from_scratch import GaussianMixture, GaussianNB, make_blobs

CENTERS = [[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]]


def _data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(300, CENTERS, cluster_std=1.2, random_state=0)
    return X, y


def test_closed_form_estimates() -> None:
    X, y = _data()
    model = GaussianNB(var_smoothing=0.0).fit(X, y)
    for k, c in enumerate(model.classes_):
        Xk = X[y == c]
        assert model.class_count_[k] == len(Xk)
        assert model.class_prior_[k] == pytest.approx(len(Xk) / len(X))
        assert np.allclose(model.theta_[k], Xk.mean(axis=0))
        assert np.allclose(model.var_[k], Xk.var(axis=0))  # MLE: divide by n_k


def test_posterior_matches_bayes_rule() -> None:
    X, y = _data()
    model = GaussianNB().fit(X, y)
    x = np.array([0.3, -0.7])
    dens = model.class_prior_ * np.prod(
        np.exp(-0.5 * (x - model.theta_) ** 2 / model.var_) / np.sqrt(2 * np.pi * model.var_),
        axis=1,
    )
    P = model.predict_proba(x[None, :])
    assert np.allclose(P[0], dens / dens.sum())
    assert np.allclose(
        model.predict_log_proba(X).sum(axis=0), np.log(model.predict_proba(X)).sum(axis=0)
    )
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_learns_blobs_and_string_labels() -> None:
    X, y = _data()
    labels = np.array(["a", "b", "c"])[y]
    model = GaussianNB().fit(X, labels)
    assert model.score(X, labels) > 0.85
    assert set(model.predict(X)) <= {"a", "b", "c"}


def test_far_away_points_do_not_underflow() -> None:
    X, y = _data()
    model = GaussianNB().fit(X, y)
    far = np.array([[1e3, 1e3], [-1e3, 5e2]])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        logp = model.predict_log_proba(far)
    assert np.all(np.isfinite(logp))
    assert np.allclose(np.exp(logp).sum(axis=1), 1.0)


def test_var_smoothing_handles_constant_feature_and_single_sample_class() -> None:
    X = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [5.0, 1.0]])
    y = np.array([0, 0, 0, 1])
    model = GaussianNB().fit(X, y)
    assert model.epsilon_ == pytest.approx(1e-9 * np.var(X[:, 0]))
    assert np.all(model.var_ > 0)
    assert np.all(np.isfinite(model.predict_log_proba(X)))
    with pytest.raises(ValueError, match="zero variance"):
        GaussianNB(var_smoothing=0.0).fit(X, y)


def test_fixed_priors_shift_the_decision() -> None:
    X, y = _data()
    uniform = GaussianNB().fit(X, y)
    skewed = GaussianNB(priors=[0.98, 0.01, 0.01]).fit(X, y)
    assert np.allclose(skewed.class_prior_, [0.98, 0.01, 0.01])
    assert np.mean(skewed.predict(X) == 0) > np.mean(uniform.predict(X) == 0)
    with pytest.raises(ValueError, match="priors"):
        GaussianNB(priors=[0.5, 0.5]).fit(X, y)
    with pytest.raises(ValueError, match="sum to one"):
        GaussianNB(priors=[0.5, 0.5, 0.5]).fit(X, y)


def test_equals_diagonal_gaussian_class_densities() -> None:
    # p(x | y=k) is a single diagonal Gaussian: a one-component diag GaussianMixture fitted
    # to the class (reg_covar = epsilon) gives the same log-density
    X, y = _data()
    model = GaussianNB().fit(X, y)
    jll = model._joint_log_likelihood(X)
    for k, c in enumerate(model.classes_):
        gm = GaussianMixture(
            1, covariance_type="diag", reg_covar=model.epsilon_, random_state=0
        ).fit(X[y == c])
        assert np.allclose(jll[:, k], np.log(model.class_prior_[k]) + gm.score_samples(X))


def test_errors() -> None:
    X, y = _data()
    with pytest.raises(RuntimeError):
        GaussianNB().predict(X)
    with pytest.raises(ValueError):
        GaussianNB(var_smoothing=-1.0)
    model = GaussianNB().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        GaussianNB().fit(X, y[:-1])


def test_matches_sklearn() -> None:
    sk = pytest.importorskip("sklearn.naive_bayes")
    X, y = _data()
    ours = GaussianNB().fit(X, y)
    ref = sk.GaussianNB().fit(X, y)
    assert np.allclose(ours.theta_, ref.theta_)
    assert np.allclose(ours.var_, ref.var_)
    assert np.allclose(ours.class_prior_, ref.class_prior_)
    assert ours.epsilon_ == pytest.approx(ref.epsilon_)
    assert np.allclose(ours.predict_log_proba(X), ref.predict_log_proba(X))
    assert np.array_equal(ours.predict(X), ref.predict(X))
