"""Multinomial (softmax) logistic regression trained with batch gradient descent."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._utils import FloatArray, as_2d_float, check_consistent_length, logsumexp, softmax


class SoftmaxRegression:
    r"""L2-regularised multinomial logistic regression for ``K >= 2`` classes.

    Model: :math:`P(y=k\mid x) = \operatorname{softmax}(z)_k = e^{z_k} / \sum_j e^{z_j}`
    with logits :math:`z = W^\top x + b`, :math:`W \in \mathbb{R}^{d\times K}`,
    :math:`b \in \mathbb{R}^K`.

    With one-hot targets :math:`Y \in \{0,1\}^{n\times K}` the objective is the mean
    cross-entropy plus a ridge term on the weights (the intercepts are not penalised):

    .. math::
        J(W, b) = \frac{1}{n}\sum_i \Big[\log\sum_j e^{z_{ij}} - \sum_k Y_{ik} z_{ik}\Big]
                  + \frac{\lambda}{2}\lVert W\rVert_F^2

    The log-sum-exp is evaluated with the row maximum subtracted, so large logits never
    overflow. The gradient generalises the binary case:

    .. math::
        \nabla_W J = \tfrac{1}{n} X^\top(P - Y) + \lambda W, \qquad
        \nabla_b J = \tfrac{1}{n}\textstyle\sum_i (P_i - Y_i), \qquad
        P = \operatorname{softmax}(XW + b)

    ``J`` is convex. Adding the same vector to every column of the logits leaves the
    probabilities unchanged, so without regularisation the parameters are not unique.
    Starting from zero, every gradient step keeps the columns of ``W`` and the entries
    of ``b`` summing to zero (each row of ``P - Y`` sums to zero), which picks the same
    symmetric solution as scikit-learn. For ``K = 2`` the model is equivalent to binary
    logistic regression with ``w = w_1 - w_0``. Equivalence with scikit-learn:
    ``C = 1 / (n_samples * l2)``.

    Args:
        learning_rate: Step size.
        n_iter: Maximum number of gradient steps.
        l2: Ridge strength ``lambda >= 0`` (the intercepts are not penalised).
        fit_intercept: Learn the biases ``b``.
        tol: Stop when the loss improves by less than ``tol``.

    Attributes:
        classes_: Sorted class labels; column ``k`` of the probabilities is ``classes_[k]``.
        coef_: Weights of shape ``(n_classes, n_features)`` (scikit-learn layout).
        intercept_: Biases of shape ``(n_classes,)``.
        loss_history_: Objective after every iteration.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.5,
        n_iter: int = 1000,
        l2: float = 0.0,
        fit_intercept: bool = True,
        tol: float = 1e-12,
    ) -> None:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if learning_rate <= 0 or n_iter < 1:
            raise ValueError("learning_rate must be positive and n_iter >= 1")
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit on ``X`` of shape ``(n, d)`` and a label vector ``y`` with at least two classes."""
        Xa = as_2d_float(X)
        ya = np.asarray(y)
        if ya.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {ya.shape}")
        check_consistent_length(Xa, ya)
        self.classes_: NDArray[np.generic] = np.unique(ya)
        if len(self.classes_) < 2:
            raise ValueError("need at least 2 classes")
        Y = (ya[:, None] == self.classes_[None, :]).astype(np.float64)

        n, d = Xa.shape
        W = np.zeros((d, len(self.classes_)))
        b = np.zeros(len(self.classes_))
        self.loss_history_: list[float] = []
        prev = np.inf
        for _ in range(self.n_iter):
            Z = Xa @ W + b
            loss = self._objective(Z, Y, W)
            self.loss_history_.append(loss)
            if abs(prev - loss) < self.tol:
                break
            prev = loss
            residual = softmax(Z) - Y
            W = W - self.learning_rate * (Xa.T @ residual / n + self.l2 * W)
            if self.fit_intercept:
                b = b - self.learning_rate * residual.mean(axis=0)
        self.coef_: FloatArray = W.T.copy()
        self.intercept_: FloatArray = b
        return self

    def _objective(self, Z: FloatArray, Y: FloatArray, W: FloatArray) -> float:
        """Mean cross-entropy of the logits ``Z`` plus ``lambda / 2 * ||W||_F^2``."""
        data = float(np.mean(logsumexp(Z) - (Y * Z).sum(axis=1)))
        return data + 0.5 * self.l2 * float(np.sum(W * W))

    def decision_function(self, X: ArrayLike) -> FloatArray:
        """Return the logits ``Z = X W^T + b`` of shape ``(n, n_classes)``."""
        self._check_fitted()
        Xa = as_2d_float(X)
        if Xa.shape[1] != self.coef_.shape[1]:
            raise ValueError(f"expected {self.coef_.shape[1]} features, got {Xa.shape[1]}")
        return Xa @ self.coef_.T + self.intercept_

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Return class probabilities of shape ``(n, n_classes)``; rows sum to one."""
        return softmax(self.decision_function(X))

    def predict_log_proba(self, X: ArrayLike) -> FloatArray:
        """Return ``log P(y = k | x)`` computed as ``z_k - logsumexp(z)`` (no underflow)."""
        Z = self.decision_function(X)
        return Z - logsumexp(Z)[:, None]

    def predict(self, X: ArrayLike) -> NDArray[np.generic]:
        """Return the most probable label for every row."""
        Z = self.decision_function(X)  # also checks that the model is fitted
        return self.classes_[np.argmax(Z, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Mean accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("call fit() before predict()")
