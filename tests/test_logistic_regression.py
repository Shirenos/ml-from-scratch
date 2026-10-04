import numpy as np
import pytest

from ml_from_scratch import LogisticRegression, make_blobs
from ml_from_scratch._utils import sigmoid


def _data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(300, [[-1.5, -1.0], [1.5, 1.0]], cluster_std=1.1, random_state=0)
    return X, y


def test_sigmoid_is_stable_and_correct() -> None:
    z = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        s = sigmoid(z)
    assert s[2] == 0.5
    assert s[0] == 0.0
    assert s[-1] == 1.0
    assert s[1] == pytest.approx(1 / (1 + np.exp(2.0)))
    assert np.allclose(s + sigmoid(-z), 1.0)


def test_separable_data_is_learned() -> None:
    X, y = make_blobs(100, [[-3.0, 0.0], [3.0, 0.0]], cluster_std=0.5, random_state=1)
    model = LogisticRegression(n_iter=500).fit(X, y)
    assert model.score(X, y) == 1.0
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_loss_decreases_monotonically() -> None:
    X, y = _data()
    model = LogisticRegression(learning_rate=0.1, n_iter=300).fit(X, y)
    assert np.all(np.diff(model.loss_history_) <= 1e-12)


def test_initial_loss_is_log_two() -> None:
    X, y = _data()
    model = LogisticRegression(n_iter=1).fit(X, y)
    assert model.loss_history_[0] == pytest.approx(np.log(2.0))


def test_first_step_follows_numerical_gradient() -> None:
    # one step from w = 0, b = 0 moves the parameters by -lr * grad J
    X, y = _data()
    lr = 0.01
    model = LogisticRegression(learning_rate=lr, n_iter=1).fit(X, y)

    def loss(theta: np.ndarray) -> float:
        z = X @ theta[:-1] + theta[-1]
        return float(np.mean(np.logaddexp(0, z) - y * z))

    eps = 1e-6
    grad = np.array([(loss(e * eps) - loss(-e * eps)) / (2 * eps) for e in np.eye(X.shape[1] + 1)])
    assert np.allclose(np.append(model.coef_, model.intercept_), -lr * grad, atol=1e-9)


def test_matches_sklearn_l2() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")
    X, y = _data()
    n, l2 = len(X), 0.05
    ours = LogisticRegression(learning_rate=0.5, n_iter=20000, l2=l2, tol=0.0).fit(X, y)
    ref = linear_model.LogisticRegression(C=1.0 / (n * l2), tol=1e-12, max_iter=10000).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_[0], atol=1e-4)
    assert ours.intercept_ == pytest.approx(ref.intercept_[0], abs=1e-4)
    assert np.allclose(ours.predict_proba(X), ref.predict_proba(X)[:, 1], atol=1e-4)
    assert np.array_equal(ours.predict(X), ref.predict(X))


def test_string_labels_and_validation() -> None:
    X, y = _data()
    labels = np.where(y == 1, "spam", "ham")
    model = LogisticRegression(n_iter=200).fit(X, labels)
    assert set(model.predict(X)) <= {"spam", "ham"}
    assert model.classes_[1] == "spam"
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, np.arange(len(X)) % 3)
    with pytest.raises(RuntimeError):
        LogisticRegression().predict(X)
