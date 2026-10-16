import numpy as np
import pytest

from ml_from_scratch import LogisticRegression, SoftmaxRegression, make_blobs
from ml_from_scratch._utils import logsumexp, softmax

CENTERS = [[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]]


def _data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(300, CENTERS, cluster_std=1.2, random_state=0)
    return X, y


def _one_hot(y: np.ndarray, k: int) -> np.ndarray:
    return np.eye(k)[y]


def test_logsumexp_and_softmax_are_stable() -> None:
    Z = np.array([[1000.0, 1000.0, -1000.0], [-1000.0, -1001.0, -1002.0], [0.0, 0.0, 0.0]])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        lse = logsumexp(Z)
        P = softmax(Z)
    assert lse[0] == pytest.approx(1000.0 + np.log(2.0))
    assert lse[1] == pytest.approx(-1000.0 + np.log(1 + np.exp(-1.0) + np.exp(-2.0)))
    assert lse[2] == pytest.approx(np.log(3.0))
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.allclose(P[0], [0.5, 0.5, 0.0])


def test_initial_loss_is_log_k() -> None:
    X, y = _data()
    model = SoftmaxRegression(n_iter=1).fit(X, y)
    assert model.loss_history_[0] == pytest.approx(np.log(3.0))


def test_loss_decreases_monotonically_and_data_is_learned() -> None:
    X, y = _data()
    model = SoftmaxRegression(learning_rate=0.2, n_iter=500).fit(X, y)
    assert np.all(np.diff(model.loss_history_) <= 1e-12)
    assert model.score(X, y) > 0.85


@pytest.mark.parametrize("l2", [0.0, 0.3])
def test_gradient_matches_finite_differences(l2: float) -> None:
    # the analytic gradient agrees with central differences at a random point, and
    # one fit step from W = 0, b = 0 moves the parameters by exactly -lr * grad J
    X, y = _data()
    k, d = 3, X.shape[1]
    Y = _one_hot(y, k)
    rng = np.random.default_rng(1)
    W0 = rng.normal(size=(d, k))
    b0 = rng.normal(size=k)

    def loss(theta: np.ndarray) -> float:
        W, b = theta[: d * k].reshape(d, k), theta[d * k :]
        Z = X @ W + b
        return float(np.mean(logsumexp(Z) - (Y * Z).sum(axis=1)) + 0.5 * l2 * np.sum(W * W))

    theta0 = np.concatenate([W0.ravel(), b0])
    eps = 1e-6
    numeric = np.array(
        [(loss(theta0 + eps * e) - loss(theta0 - eps * e)) / (2 * eps) for e in np.eye(len(theta0))]
    )
    R = softmax(X @ W0 + b0) - Y
    analytic = np.concatenate([(X.T @ R / len(X) + l2 * W0).ravel(), R.mean(axis=0)])
    assert np.allclose(analytic, numeric, atol=1e-7)

    lr = 0.01
    model = SoftmaxRegression(learning_rate=lr, n_iter=1, l2=l2).fit(X, y)
    R0 = softmax(np.zeros((len(X), k))) - Y
    step = -lr * np.concatenate([(X.T @ R0 / len(X)).ravel(), R0.mean(axis=0)])
    assert np.allclose(np.concatenate([model.coef_.T.ravel(), model.intercept_]), step)


def test_intercept_is_not_penalised() -> None:
    # stationarity in b gives mean(P) = mean(Y): the average predicted probabilities
    # match the class frequencies exactly, even when a strong l2 shrinks W towards 0
    X, y = make_blobs(400, CENTERS, cluster_std=1.0, random_state=3)
    keep = np.concatenate([np.flatnonzero(y == 0), np.flatnonzero(y != 0)[::3]])
    X, y = X[keep], y[keep]
    model = SoftmaxRegression(learning_rate=0.01, n_iter=5000, l2=50.0, tol=0.0).fit(X, y)
    assert np.abs(model.coef_).max() < 0.05
    priors = np.bincount(y) / len(y)
    assert np.allclose(model.predict_proba(X).mean(axis=0), priors, atol=1e-8)
    assert not np.allclose(priors, 1 / 3, atol=0.05)


def test_parameters_stay_centred_across_classes() -> None:
    X, y = _data()
    model = SoftmaxRegression(n_iter=300).fit(X, y)
    assert np.allclose(model.coef_.sum(axis=0), 0.0, atol=1e-10)
    assert model.intercept_.sum() == pytest.approx(0.0, abs=1e-10)


def test_two_classes_match_binary_logistic_regression() -> None:
    X, y = make_blobs(200, [[-1.5, -1.0], [1.5, 1.0]], cluster_std=1.1, random_state=0)
    l2 = 0.02
    soft = SoftmaxRegression(learning_rate=0.5, n_iter=20000, l2=l2, tol=0.0).fit(X, y)
    # the symmetric softmax solution has w_1 = -w_0 = w / 2, and its penalty
    # lambda/2 * (|w_0|^2 + |w_1|^2) = (lambda/2) / 2 * |w|^2, i.e. binary strength l2 / 2
    binary = LogisticRegression(learning_rate=0.5, n_iter=20000, l2=l2 / 2, tol=0.0).fit(X, y)
    assert np.allclose(soft.coef_[1] - soft.coef_[0], binary.coef_, atol=1e-5)
    assert np.allclose(soft.predict_proba(X)[:, 1], binary.predict_proba(X), atol=1e-6)


def test_matches_sklearn_multinomial_l2() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")
    X, y = _data()
    n, l2 = len(X), 0.05
    ours = SoftmaxRegression(learning_rate=0.5, n_iter=20000, l2=l2, tol=0.0).fit(X, y)
    ref = linear_model.LogisticRegression(C=1.0 / (n * l2), tol=1e-12, max_iter=10000).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-4)
    assert np.allclose(ours.intercept_, ref.intercept_, atol=1e-4)
    assert np.allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-4)
    assert np.array_equal(ours.predict(X), ref.predict(X))


def test_log_proba_is_finite_for_extreme_logits() -> None:
    X, y = make_blobs(60, CENTERS, cluster_std=0.3, random_state=2)
    model = SoftmaxRegression(learning_rate=1.0, n_iter=2000).fit(X, y)
    far = np.array([[1e4, -1e4]])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        log_p = model.predict_log_proba(far)
    assert np.all(np.isfinite(log_p))
    assert np.allclose(np.exp(log_p), model.predict_proba(far))


def test_string_labels_and_validation() -> None:
    X, y = _data()
    labels = np.array(["cat", "dog", "fox"])[y]
    model = SoftmaxRegression(n_iter=200).fit(X, labels)
    assert list(model.classes_) == ["cat", "dog", "fox"]
    assert set(model.predict(X)) <= {"cat", "dog", "fox"}
    assert model.predict_proba(X).shape == (len(X), 3)
    with pytest.raises(ValueError):
        SoftmaxRegression().fit(X, np.zeros(len(X)))
    with pytest.raises(ValueError):
        SoftmaxRegression(l2=-1.0)
    with pytest.raises(RuntimeError):
        SoftmaxRegression().predict(X)
    with pytest.raises(ValueError):
        model.predict(X[:, :1])
