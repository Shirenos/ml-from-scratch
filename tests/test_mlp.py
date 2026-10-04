import numpy as np
import pytest

from ml_from_scratch import MLP, make_blobs, make_moons, train_test_split
from ml_from_scratch.mlp import Task


def _numeric_grad(model: MLP, X: np.ndarray, Y: np.ndarray, param: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(param)
    eps = 1e-6
    for idx in np.ndindex(*param.shape):
        old = param[idx]
        param[idx] = old + eps
        plus = model._loss_from_logits(model._forward(X)[0][-1], Y)
        param[idx] = old - eps
        minus = model._loss_from_logits(model._forward(X)[0][-1], Y)
        param[idx] = old
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_backprop_matches_numerical_gradient(task: Task, activation: str) -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 3))
    if task == "binary":
        y = rng.integers(0, 2, 12)
    elif task == "multiclass":
        y = np.arange(12) % 3
    else:
        y = rng.normal(size=(12, 2))
    model = MLP((5, 4), task=task, activation=activation, l2=0.01, n_epochs=1, random_state=0)  # type: ignore[arg-type]
    model.fit(X, y)  # builds the parameters (one epoch)
    Y = model._encode_targets(np.asarray(y))
    _, gW, gb = model._loss_and_grads(X, Y)
    for W, g in zip(model.weights_, gW, strict=True):
        assert np.allclose(g, _numeric_grad(model, X, Y, W), atol=1e-6)
    for b, g in zip(model.biases_, gb, strict=True):
        assert np.allclose(g, _numeric_grad(model, X, Y, b), atol=1e-6)


def test_learns_xor() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 8, dtype=float)
    y = np.array([0, 1, 1, 0] * 8)
    model = MLP((8,), learning_rate=0.05, n_epochs=600, random_state=1).fit(X, y)
    assert model.score(X, y) == 1.0
    assert model.loss_history_[-1] < 0.05


def test_moons_generalises() -> None:
    X, y = make_moons(400, noise=0.15, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    model = MLP((16, 16), learning_rate=0.02, n_epochs=300, batch_size=64, random_state=0)
    model.fit(Xtr, ytr)
    assert model.score(Xte, yte) > 0.92
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_multiclass_blobs_and_probabilities() -> None:
    X, y = make_blobs(240, [[-4, 0], [0, 4], [4, 0]], cluster_std=0.7, random_state=0)
    model = MLP((10,), task="multiclass", learning_rate=0.02, n_epochs=200, random_state=0)
    model.fit(X, y)
    assert model.score(X, y) > 0.97
    proba = model.predict_proba(X)
    assert proba.shape == (240, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_regression_fits_sine_with_sgd() -> None:
    x = np.linspace(-3, 3, 120)[:, None]
    y = np.sin(x[:, 0])
    model = MLP(
        (24,), task="regression", optimizer="sgd", learning_rate=0.05, n_epochs=4000, random_state=0
    ).fit(x, y)
    assert model.score(x, y) > 0.98


def test_reproducible_with_seed() -> None:
    X, y = make_moons(100, noise=0.2, random_state=0)
    a = MLP((6,), n_epochs=20, batch_size=16, random_state=5).fit(X, y)
    b = MLP((6,), n_epochs=20, batch_size=16, random_state=5).fit(X, y)
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))


def test_validation() -> None:
    X, y = make_moons(20, random_state=0)
    with pytest.raises(RuntimeError):
        MLP().predict(X)
    with pytest.raises(ValueError):
        MLP(task="binary").fit(X, np.arange(20) % 3)
    with pytest.raises(ValueError):
        MLP(activation="sigmoid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MLP((0,))
    model = MLP((4,), n_epochs=2).fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.zeros((3, 5)))
