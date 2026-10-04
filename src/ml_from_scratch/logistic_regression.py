"""Binary logistic regression trained with batch gradient descent."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._utils import (
    FloatArray,
    as_2d_float,
    check_consistent_length,
    sigmoid,
    softplus,
)


class LogisticRegression:
    r"""L2-regularised binary logistic regression.

    Model: :math:`P(y=1\mid x) = \sigma(w^\top x + b)` with
    :math:`\sigma(z) = 1/(1+e^{-z})`.

    The objective is the mean negative log-likelihood (cross-entropy) plus a ridge term:

    .. math::
        J(w, b) = \frac{1}{n}\sum_i \big[\log(1 + e^{z_i}) - y_i z_i\big]
                  + \frac{\lambda}{2}\lVert w\rVert^2,
        \qquad z_i = w^\top x_i + b

    where :math:`\log(1+e^{z}) - y z = -y\log\sigma(z) - (1-y)\log(1-\sigma(z))`
    is evaluated stably. The gradient has the remarkably simple form

    .. math::
        \nabla_w J = \tfrac{1}{n} X^\top(\sigma(z) - y) + \lambda w, \qquad
        \nabla_b J = \tfrac{1}{n}\sum_i(\sigma(z_i) - y_i)

    ``J`` is convex, so gradient descent reaches the global optimum. There is no closed
    form. Equivalence with scikit-learn: ``C = 1 / (n_samples * l2)``.

    Args:
        learning_rate: Step size.
        n_iter: Maximum number of gradient steps.
        l2: Ridge strength ``lambda >= 0`` (the intercept is not penalised).
        fit_intercept: Learn the bias ``b``.
        tol: Stop when the loss improves by less than ``tol``.

    Attributes:
        classes_: The two class labels; ``classes_[1]`` is the "positive" class.
        coef_: Weights of shape ``(n_features,)``.
        intercept_: Bias ``b``.
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

    def fit(self, X: ArrayLike, y: ArrayLike) -> LogisticRegression:
        """Fit on ``X`` of shape ``(n, d)`` and a vector ``y`` with exactly two distinct labels."""
        Xa = as_2d_float(X)
        ya = np.asarray(y)
        if ya.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {ya.shape}")
        check_consistent_length(Xa, ya)
        self.classes_: NDArray[np.generic] = np.unique(ya)
        if len(self.classes_) != 2:
            raise ValueError(f"expected exactly 2 classes, got {len(self.classes_)}")
        t = (ya == self.classes_[1]).astype(np.float64)

        n, d = Xa.shape
        w = np.zeros(d)
        b = 0.0
        self.loss_history_: list[float] = []
        prev = np.inf
        for _ in range(self.n_iter):
            z = Xa @ w + b
            loss = float(np.mean(softplus(z) - t * z)) + 0.5 * self.l2 * float(w @ w)
            self.loss_history_.append(loss)
            if abs(prev - loss) < self.tol:
                break
            prev = loss
            residual = sigmoid(z) - t
            w = w - self.learning_rate * (Xa.T @ residual / n + self.l2 * w)
            if self.fit_intercept:
                b -= self.learning_rate * float(residual.mean())
        self.coef_: FloatArray = w
        self.intercept_: float = b
        return self

    def decision_function(self, X: ArrayLike) -> FloatArray:
        """Return the logits ``z = X w + b``."""
        self._check_fitted()
        Xa = as_2d_float(X)
        if Xa.shape[1] != self.coef_.shape[0]:
            raise ValueError(f"expected {self.coef_.shape[0]} features, got {Xa.shape[1]}")
        return Xa @ self.coef_ + self.intercept_

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Return ``P(y = classes_[1] | x)`` as a vector of shape ``(n,)``."""
        return sigmoid(self.decision_function(X))

    def predict(self, X: ArrayLike) -> NDArray[np.generic]:
        """Return the predicted labels (threshold 0.5, i.e. ``z >= 0``)."""
        z = self.decision_function(X)  # also checks that the model is fitted
        return self.classes_[(z >= 0).astype(np.intp)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Mean accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("call fit() before predict()")
