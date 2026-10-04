from typing import Any

import numpy as np
import pytest

from ml_from_scratch import MLP, make_blobs, make_moons, train_test_split
from ml_from_scratch.mlp import Task


def _numeric_grad(model: MLP, X: np.ndarray, Y: np.ndarray, param: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(param)
    eps = 1e-6
    # ReLU is not differentiable at 0, so a central difference could straddle the kink and
    # disagree with the analytic gradient. With the fixed seeds below no pre-activation falls
    # within eps of 0, which keeps this check deterministic and stable.
    for idx in np.ndindex(*param.shape):
        old = param[idx]
        param[idx] = old + eps
        plus = model._loss_from_logits(model._logits(X), Y)
        param[idx] = old - eps
        minus = model._loss_from_logits(model._logits(X), Y)
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


def test_regression_fits_sine_with_adam() -> None:
    x = np.linspace(-3, 3, 120)[:, None]
    y = np.sin(x[:, 0])
    model = MLP((24,), task="regression", learning_rate=0.02, n_epochs=500, random_state=0)
    model.fit(x, y)
    assert model.score(x, y) > 0.99


def test_sgd_reduces_regression_loss() -> None:
    x = np.linspace(-3, 3, 60)[:, None]
    y = np.sin(x[:, 0])
    model = MLP(
        (12,), task="regression", optimizer="sgd", learning_rate=0.05, n_epochs=300, random_state=0
    ).fit(x, y)
    assert model.loss_history_[-1] < 0.5 * model.loss_history_[0]


def test_reproducible_with_seed() -> None:
    X, y = make_moons(100, noise=0.2, random_state=0)
    a = MLP((6,), n_epochs=20, batch_size=16, random_state=5).fit(X, y)
    b = MLP((6,), n_epochs=20, batch_size=16, random_state=5).fit(X, y)
    assert np.allclose(a.predict_proba(X), b.predict_proba(X), rtol=0, atol=1e-12)


class _StepCounter(MLP):
    """MLP that counts optimiser updates, to check how the data is split into batches."""

    n_updates = 0

    def _apply_update(self, *args: Any, **kwargs: Any) -> None:
        self.n_updates += 1
        super()._apply_update(*args, **kwargs)


@pytest.mark.parametrize(
    ("n", "batch_size", "updates_per_epoch"),
    [
        (30, 1000, 1),  # batch larger than the data -> one full batch
        (30, 30, 1),
        (25, 10, 3),  # 10 + 10 + 5: incomplete last batch
        (25, 24, 2),  # 24 + 1: last batch of a single sample
        (24, 8, 3),  # evenly divisible
    ],
)
def test_batching_update_counts(n: int, batch_size: int, updates_per_epoch: int) -> None:
    X, y = make_moons(n, noise=0.2, random_state=0)
    model = _StepCounter((4,), n_epochs=3, batch_size=batch_size).fit(X, y)
    assert model.n_updates == 3 * updates_per_epoch
    assert len(model.loss_history_) == 3
    assert np.all(np.isfinite(model.loss_history_))


def test_oversized_batch_equals_full_batch() -> None:
    X, y = make_moons(30, noise=0.2, random_state=0)
    full = MLP((4,), n_epochs=10, batch_size=None, random_state=2).fit(X, y)
    big = MLP((4,), n_epochs=10, batch_size=1000, random_state=2).fit(X, y)
    for a, b in zip(full.weights_, big.weights_, strict=True):
        assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_incomplete_last_batch_still_learns() -> None:
    X, y = make_moons(205, noise=0.1, random_state=0)  # 205 = 3 * 64 + 13
    model = MLP((16,), learning_rate=0.02, n_epochs=150, batch_size=64, random_state=0)
    model.fit(X, y)
    assert model.score(X, y) > 0.9


def test_l2_shrinks_weights() -> None:
    X, y = make_moons(100, noise=0.2, random_state=0)

    def weight_norm(l2: float) -> float:
        model = MLP((16,), n_epochs=200, l2=l2, random_state=0).fit(X, y)
        return float(np.sqrt(sum((W * W).sum() for W in model.weights_)))

    assert weight_norm(0.1) < weight_norm(0.0)


def test_binary_predict_proba_shape_and_range() -> None:
    X, y = make_moons(40, random_state=0)
    model = MLP((4,), n_epochs=5).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (40,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert model.forward(X).shape == (40, 1)


def test_multi_output_regression_shapes() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    Y = np.column_stack([X[:, 0] + X[:, 1], X[:, 2] ** 2, np.ones(50)])
    model = MLP((8,), task="regression", n_epochs=300, learning_rate=0.02).fit(X, Y)
    assert model.predict(X).shape == (50, 3)
    assert model.forward(X).shape == (50, 3)
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert model.score(X, Y) > 0.5


def test_regression_score_is_r2() -> None:
    x = np.linspace(-1, 1, 40)[:, None]
    y = 2.0 * x[:, 0]
    model = MLP((8,), task="regression", learning_rate=0.02, n_epochs=300).fit(x, y)
    pred = model.predict(x)
    expected = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert model.score(x, y) == pytest.approx(expected)
    assert model.score(x, y) > 0.9


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"activation": "sigmoid"}, "unknown activation 'sigmoid'"),
        ({"task": "ranking"}, "unknown task 'ranking'"),
        ({"optimizer": "rmsprop"}, "unknown optimizer 'rmsprop'"),
        ({"hidden_layers": (0,)}, "hidden layer sizes must be positive"),
        ({"hidden_layers": (4, -1)}, "hidden layer sizes must be positive"),
        ({"learning_rate": 0.0}, "learning_rate > 0"),
        ({"n_epochs": 0}, "n_epochs >= 1"),
        ({"l2": -0.1}, "l2 >= 0"),
        ({"batch_size": 0}, "batch_size must be positive"),
    ],
)
def test_constructor_rejects_bad_arguments(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MLP(**kwargs)


def test_predict_before_fit_raises() -> None:
    X, _ = make_moons(20, random_state=0)
    with pytest.raises(RuntimeError, match=r"call fit\(\) before predict\(\)"):
        MLP().predict(X)


def test_predict_proba_rejects_regression() -> None:
    X, y = make_moons(20, random_state=0)
    model = MLP((4,), task="regression", n_epochs=2).fit(X, y.astype(float))
    with pytest.raises(RuntimeError, match="only available for classification"):
        model.predict_proba(X)


def test_binary_task_rejects_three_classes() -> None:
    X, _ = make_moons(20, random_state=0)
    with pytest.raises(ValueError, match="binary task needs exactly 2 classes, got 3"):
        MLP(task="binary").fit(X, np.arange(20) % 3)


def test_multiclass_task_rejects_single_class() -> None:
    X, _ = make_moons(20, random_state=0)
    with pytest.raises(ValueError, match="multiclass task needs at least 2 classes"):
        MLP(task="multiclass").fit(X, np.zeros(20, dtype=int))


def test_predict_rejects_wrong_feature_count() -> None:
    X, y = make_moons(20, random_state=0)
    model = MLP((4,), n_epochs=2).fit(X, y)
    with pytest.raises(ValueError, match="expected 2 features, got 5"):
        model.predict(np.zeros((3, 5)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fit_rejects_non_finite_X(bad: float) -> None:
    X, y = make_moons(20, random_state=0)
    X[3, 1] = bad
    with pytest.raises(ValueError, match="X contains NaN or infinity"):
        MLP((4,), n_epochs=2).fit(X, y)


def test_predict_rejects_nan_X() -> None:
    X, y = make_moons(20, random_state=0)
    model = MLP((4,), n_epochs=2).fit(X, y)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="X contains NaN or infinity"):
        model.predict(X)


def test_fit_rejects_1d_X() -> None:
    _, y = make_moons(20, random_state=0)
    with pytest.raises(ValueError, match=r"X must be 2-D \(n_samples, n_features\)"):
        MLP(n_epochs=2).fit(np.zeros(20), y)


def test_fit_rejects_length_mismatch() -> None:
    X, y = make_moons(20, random_state=0)
    with pytest.raises(ValueError, match="X has 20 samples but y has 19"):
        MLP(n_epochs=2).fit(X, y[:-1])


def test_classification_rejects_2d_y() -> None:
    X, y = make_moons(20, random_state=0)
    with pytest.raises(ValueError, match=r"y must be 1-D for classification; got shape \(20, 1\)"):
        MLP(n_epochs=2).fit(X, y[:, None])


@pytest.mark.parametrize(
    "y",
    [np.zeros((20, 2, 2)), np.r_[np.nan, np.zeros(19)], np.r_[np.inf, np.zeros(19)]],
    ids=["3d", "nan", "inf"],
)
def test_regression_rejects_bad_targets(y: np.ndarray) -> None:
    X, _ = make_moons(20, random_state=0)
    with pytest.raises(ValueError, match="regression targets must be finite"):
        MLP(task="regression", n_epochs=2).fit(X, y)
