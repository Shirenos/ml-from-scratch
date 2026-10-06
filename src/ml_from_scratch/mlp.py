"""A small multilayer perceptron with hand-written backpropagation."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._utils import (
    FloatArray,
    as_2d_float,
    check_consistent_length,
    check_random_state,
    logsumexp,
    r2_score,
    sigmoid,
    softmax,
    softplus,
)

Task = Literal["binary", "multiclass", "regression"]
Activation = Literal["tanh", "relu"]
Optimizer = Literal["adam", "sgd"]


def _act(name: Activation, z: FloatArray) -> FloatArray:
    return np.tanh(z) if name == "tanh" else np.maximum(z, 0.0)


def _act_grad(name: Activation, z: FloatArray, a: FloatArray) -> FloatArray:
    """Derivative of the activation, given pre-activation ``z`` and activation ``a``."""
    if name == "tanh":
        return 1.0 - a * a
    return (z > 0.0).astype(np.float64)  # ReLU'(0) = 0 by convention


def clip_by_global_norm(grads: Sequence[FloatArray], max_norm: float) -> list[FloatArray]:
    r"""Rescale ``grads`` so that their joint L2 norm is at most ``max_norm``.

    With :math:`g = \sqrt{\sum_i \lVert g_i \rVert^2}` taken over all arrays, every
    array is multiplied by :math:`\min(1, \mathrm{max\_norm} / g)`. The direction of
    the full gradient is preserved; only its length is capped. Inputs are not modified.
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive; got {max_norm}")
    total = float(np.sqrt(sum(float((g * g).sum()) for g in grads)))
    if total <= max_norm:
        return list(grads)
    scale = max_norm / total
    return [g * scale for g in grads]


class MLP:
    r"""Fully connected feed-forward network trained with mini-batch gradient methods.

    **Forward pass.** With :math:`a^{(0)} = x`, for layers :math:`l = 1..L`:

    .. math::
        z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}, \qquad a^{(l)} = \phi(z^{(l)})

    for hidden layers (:math:`\phi` = tanh or ReLU); the output layer applies the link
    that matches the task, and the loss is its negative log-likelihood:

    ========== ================== ==========================================
    task       output :math:`a^{(L)}`  loss per sample
    ========== ================== ==========================================
    binary     :math:`\sigma(z)`   :math:`-y\log a-(1-y)\log(1-a)`
    multiclass softmax(:math:`z`)  :math:`-\sum_k y_k \log a_k` (one-hot y)
    regression :math:`z`           :math:`\tfrac12\lVert a - y\rVert^2`
    ========== ================== ==========================================

    **Backpropagation.** For all three pairs (link, loss) the output error is simply
    :math:`\delta^{(L)} = (a^{(L)} - y)/n`. It is propagated backwards with the chain
    rule:

    .. math::
        \delta^{(l-1)} = \big(\delta^{(l)} W^{(l)\top}\big) \odot \phi'(z^{(l-1)}), \qquad
        \frac{\partial J}{\partial W^{(l)}} = a^{(l-1)\top}\delta^{(l)} + \lambda W^{(l)}, \qquad
        \frac{\partial J}{\partial b^{(l)}} = \sum_i \delta^{(l)}_i

    Weights use Glorot (tanh) or He (ReLU) normal initialisation. Updates are plain
    SGD or Adam, optionally after clipping the gradient by its global norm.

    Args:
        hidden_layers: Sizes of the hidden layers, e.g. ``(16, 16)``.
        activation: ``"tanh"`` or ``"relu"``.
        task: ``"binary"``, ``"multiclass"`` or ``"regression"``.
        optimizer: ``"adam"`` or ``"sgd"``.
        learning_rate: Step size.
        n_epochs: Passes over the training data.
        batch_size: Mini-batch size; ``None`` means full batch.
        l2: Penalty strength ``lambda >= 0`` on the weights (not on biases). This is classic
            L2 regularisation: ``lambda * W`` is added to the gradient, so with Adam it is
            rescaled by the adaptive step. It is not decoupled weight decay (AdamW).
        clip_norm: If set, rescale each mini-batch gradient so that its global L2 norm
            (over all weights and biases together) is at most ``clip_norm``. ``None``
            disables clipping.
        random_state: Seed (or generator) for initialisation and shuffling.

    Attributes:
        weights_: List of weight matrices ``W^(l)`` of shape ``(n_in, n_out)``.
        biases_: List of bias vectors ``b^(l)``.
        loss_history_: Full-data objective after every epoch.
        classes_: Class labels (classification tasks only).
    """

    def __init__(
        self,
        hidden_layers: Sequence[int] = (16, 16),
        *,
        activation: Activation = "tanh",
        task: Task = "binary",
        optimizer: Optimizer = "adam",
        learning_rate: float = 0.01,
        n_epochs: int = 500,
        batch_size: int | None = None,
        l2: float = 0.0,
        clip_norm: float | None = None,
        random_state: int | np.random.Generator | None = 0,
    ) -> None:
        if activation not in ("tanh", "relu"):
            raise ValueError(f"unknown activation {activation!r}")
        if task not in ("binary", "multiclass", "regression"):
            raise ValueError(f"unknown task {task!r}")
        if optimizer not in ("adam", "sgd"):
            raise ValueError(f"unknown optimizer {optimizer!r}")
        if any(h < 1 for h in hidden_layers):
            raise ValueError("hidden layer sizes must be positive")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive; got {learning_rate}")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be at least 1; got {n_epochs}")
        if l2 < 0:
            raise ValueError(f"l2 must be non-negative; got {l2}")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        if clip_norm is not None and clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive; got {clip_norm}")
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.task = task
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.clip_norm = clip_norm
        self.random_state = random_state

    # ------------------------------------------------------------ initialisation
    def _init_params(self, n_in: int, n_out: int, rng: np.random.Generator) -> None:
        sizes = [n_in, *self.hidden_layers, n_out]
        self.weights_: list[FloatArray] = []
        self.biases_: list[FloatArray] = []
        for fan_in, fan_out in pairwise(sizes):
            if self.activation == "relu":
                std = np.sqrt(2.0 / fan_in)  # He
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))  # Glorot
            self.weights_.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.biases_.append(np.zeros(fan_out))

    # ------------------------------------------------------------- forward/loss
    def _forward(self, X: FloatArray) -> tuple[list[FloatArray], list[FloatArray]]:
        """Return pre-activations ``zs`` and activations ``acts`` (``acts[0] = X``).

        ``acts[-1]`` is the network output after the task-specific link.
        """
        acts = [X]
        zs: list[FloatArray] = []
        last = len(self.weights_) - 1
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_, strict=True)):
            z = acts[-1] @ W + b
            zs.append(z)
            if i < last:
                acts.append(_act(self.activation, z))
            elif self.task == "binary":
                acts.append(sigmoid(z))
            elif self.task == "multiclass":
                acts.append(softmax(z))
            else:
                acts.append(z)
        return zs, acts

    def _logits(self, X: FloatArray) -> FloatArray:
        """Pre-link output ``z^(L)`` of the last layer, shape ``(n, n_outputs)``."""
        return self._forward(X)[0][-1]

    def _loss_from_logits(self, z: FloatArray, Y: FloatArray) -> float:
        if self.task == "binary":
            data = np.mean(softplus(z) - Y * z)
        elif self.task == "multiclass":
            data = np.mean(logsumexp(z) - (Y * z).sum(axis=1))
        else:
            data = 0.5 * np.mean(((z - Y) ** 2).sum(axis=1))
        penalty = 0.5 * self.l2 * sum(float((W * W).sum()) for W in self.weights_)
        return float(data) + penalty

    def _grads(self, X: FloatArray, Y: FloatArray) -> tuple[list[FloatArray], list[FloatArray]]:
        """Gradients of the objective w.r.t. every weight matrix and bias vector."""
        n = X.shape[0]
        zs, acts = self._forward(X)
        delta = (acts[-1] - Y) / n  # output error for all (link, loss) pairs above
        grads_W: list[FloatArray] = [np.empty(0)] * len(self.weights_)
        grads_b: list[FloatArray] = [np.empty(0)] * len(self.weights_)
        for layer in range(len(self.weights_) - 1, -1, -1):
            grads_W[layer] = acts[layer].T @ delta + self.l2 * self.weights_[layer]
            grads_b[layer] = delta.sum(axis=0)
            if layer > 0:
                back = delta @ self.weights_[layer].T
                delta = back * _act_grad(self.activation, zs[layer - 1], acts[layer])
        return grads_W, grads_b

    # ---------------------------------------------------------------- training
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Train on ``X`` of shape ``(n, d)`` and targets ``y``.

        ``y`` holds class labels (classification) or real values, shape ``(n,)`` or
        ``(n, k)`` (regression).
        """
        Xa = as_2d_float(X)
        ya = np.asarray(y)
        check_consistent_length(Xa, ya)
        Y = self._encode_targets(ya)
        rng = check_random_state(self.random_state)
        self._init_params(Xa.shape[1], Y.shape[1], rng)

        params = [*self.weights_, *self.biases_]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        step = 0
        n = Xa.shape[0]
        bs = n if self.batch_size is None else min(self.batch_size, n)
        self.loss_history_: list[float] = []
        for _ in range(self.n_epochs):
            order = rng.permutation(n)
            for start in range(0, n, bs):
                idx = order[start : start + bs]
                gW, gb = self._grads(Xa[idx], Y[idx])
                grads = [*gW, *gb]
                if self.clip_norm is not None:
                    grads = clip_by_global_norm(grads, self.clip_norm)
                step += 1
                self._apply_update(params, grads, m, v, step)
            self.loss_history_.append(self._loss_from_logits(self._logits(Xa), Y))
        return self

    def _apply_update(
        self,
        params: list[FloatArray],
        grads: list[FloatArray],
        m: list[FloatArray],
        v: list[FloatArray],
        step: int,
    ) -> None:
        """In-place SGD / Adam update (parameter arrays are shared with the model)."""
        lr = self.learning_rate
        if self.optimizer == "sgd":
            for p, g in zip(params, grads, strict=True):
                p -= lr * g
            return
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        c1, c2 = 1.0 - beta1**step, 1.0 - beta2**step
        for p, g, mi, vi in zip(params, grads, m, v, strict=True):
            mi *= beta1
            mi += (1.0 - beta1) * g
            vi *= beta2
            vi += (1.0 - beta2) * g * g
            p -= lr * (mi / c1) / (np.sqrt(vi / c2) + eps)

    def _encode_targets(self, y: NDArray[np.generic]) -> FloatArray:
        if self.task == "regression":
            Y = np.asarray(y, dtype=np.float64)
            if Y.ndim == 1:
                Y = Y[:, None]
            if Y.ndim != 2 or not np.all(np.isfinite(Y)):
                raise ValueError("regression targets must be finite with shape (n,) or (n, k)")
            return Y
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D for classification; got shape {y.shape}")
        self.classes_: NDArray[np.generic] = np.unique(y)
        if self.task == "binary":
            if len(self.classes_) != 2:
                raise ValueError(f"binary task needs exactly 2 classes, got {len(self.classes_)}")
            return np.asarray(y == self.classes_[1], dtype=np.float64)[:, None]
        if len(self.classes_) < 2:
            raise ValueError("multiclass task needs at least 2 classes")
        return np.asarray(y[:, None] == self.classes_[None, :], dtype=np.float64)

    # -------------------------------------------------------------- prediction
    def forward(self, X: ArrayLike) -> FloatArray:
        """Network output ``a^(L)`` of shape ``(n, n_outputs)``."""
        self._check_fitted()
        Xa = as_2d_float(X)
        if Xa.shape[1] != self.weights_[0].shape[0]:
            raise ValueError(f"expected {self.weights_[0].shape[0]} features, got {Xa.shape[1]}")
        return self._forward(Xa)[1][-1]

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Class probabilities: shape ``(n,)`` = P(classes_[1]) for binary, ``(n, K)`` otherwise."""
        if self.task == "regression":
            raise RuntimeError("predict_proba is only available for classification")
        out = self.forward(X)
        return out[:, 0] if self.task == "binary" else out

    def predict(self, X: ArrayLike) -> NDArray[np.generic]:
        """Predicted labels (classification) or values (regression)."""
        out = self.forward(X)
        if self.task == "regression":
            return out[:, 0] if out.shape[1] == 1 else out
        if self.task == "binary":
            return self.classes_[(out[:, 0] >= 0.5).astype(np.intp)]
        return self.classes_[np.argmax(out, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Accuracy for classification, R^2 for regression."""
        if self.task == "regression":
            return r2_score(np.asarray(y, dtype=np.float64), np.asarray(self.predict(X)))
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _check_fitted(self) -> None:
        if not hasattr(self, "weights_"):
            raise RuntimeError("call fit() before predict()")
