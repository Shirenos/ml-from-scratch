"""Linear regression: closed-form least squares and batch gradient descent.

Model: ``y_hat = X w + b`` with the objective

    J(w, b) = 1/(2n) * ||X w + b - y||^2  +  (l2 / 2) * ||w||^2

(the intercept ``b`` is never penalised).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._utils import (
    FloatArray,
    as_1d_float,
    as_2d_float,
    check_consistent_length,
    r2_score,
)

Method = Literal["closed_form", "gradient_descent"]


class LinearRegression:
    r"""Ordinary / ridge least squares fitted in closed form or by gradient descent.

    **Closed form.** Setting :math:`\nabla J = 0` gives the normal equations

    .. math:: (X^\top X + n\lambda I)\, w = X^\top y

    (:math:`\lambda` = ``l2``). On centred data the intercept is
    :math:`b = \bar y - \bar x^\top w`. For ``l2 == 0`` the system is solved with
    ``numpy.linalg.lstsq`` (SVD based, minimum-norm solution if ``X`` is rank deficient)
    instead of explicitly inverting :math:`X^\top X`.

    **Gradient descent.** With residual :math:`r = Xw + b - y`:

    .. math::
        \nabla_w J = \tfrac{1}{n} X^\top r + \lambda w, \qquad
        \nabla_b J = \tfrac{1}{n} \sum_i r_i, \qquad
        w \leftarrow w - \eta\, \nabla_w J

    The objective is a convex quadratic, so a small enough learning rate :math:`\eta`
    converges to the closed-form solution. Standardising features helps a lot.

    Equivalence with scikit-learn's ``Ridge``: ``alpha = n_samples * l2``.

    Args:
        method: ``"closed_form"`` or ``"gradient_descent"``.
        fit_intercept: Learn the bias ``b``; otherwise ``b = 0``.
        l2: Ridge penalty ``lambda >= 0``.
        learning_rate: Step size ``eta`` (gradient descent only).
        n_iter: Maximum number of iterations (gradient descent only).
        tol: Stop when the loss improves by less than ``tol`` (gradient descent only).

    Attributes:
        coef_: Weight vector ``w`` of shape ``(n_features,)``.
        intercept_: Bias ``b``.
        loss_history_: Objective ``J`` after every iteration (empty for ``closed_form``).
        n_iter_: Iterations actually performed (gradient descent only, else 0).
    """

    def __init__(
        self,
        method: Method = "closed_form",
        *,
        fit_intercept: bool = True,
        l2: float = 0.0,
        learning_rate: float = 0.1,
        n_iter: int = 1000,
        tol: float = 1e-12,
    ) -> None:
        if method not in ("closed_form", "gradient_descent"):
            raise ValueError(f"unknown method {method!r}")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if learning_rate <= 0 or n_iter < 1:
            raise ValueError("learning_rate must be positive and n_iter >= 1")
        self.method = method
        self.fit_intercept = fit_intercept
        self.l2 = l2
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tol = tol

    # ------------------------------------------------------------------ fitting
    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearRegression:
        """Fit the model to ``X`` of shape ``(n, d)`` and targets ``y`` of shape ``(n,)``."""
        Xa = as_2d_float(X)
        ya = as_1d_float(y)
        check_consistent_length(Xa, ya)
        self.loss_history_: list[float] = []
        self.n_iter_ = 0
        if self.method == "closed_form":
            self._fit_closed_form(Xa, ya)
        else:
            self._fit_gradient_descent(Xa, ya)
        return self

    def _fit_closed_form(self, X: FloatArray, y: FloatArray) -> None:
        n, d = X.shape
        if self.fit_intercept:
            x_mean, y_mean = X.mean(axis=0), y.mean()
            Xc, yc = X - x_mean, y - y_mean
        else:
            Xc, yc = X, y
        if self.l2 > 0:
            gram = Xc.T @ Xc + n * self.l2 * np.eye(d)
            w = np.linalg.solve(gram, Xc.T @ yc)
        else:
            w = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        self.coef_: FloatArray = w
        self.intercept_: float = float(y_mean - x_mean @ w) if self.fit_intercept else 0.0

    def _fit_gradient_descent(self, X: FloatArray, y: FloatArray) -> None:
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        prev = np.inf
        for it in range(1, self.n_iter + 1):
            r = X @ w + b - y
            loss = float(r @ r) / (2 * n) + 0.5 * self.l2 * float(w @ w)
            self.loss_history_.append(loss)
            if not np.isfinite(loss):
                raise FloatingPointError("loss diverged; lower learning_rate or scale features")
            if abs(prev - loss) < self.tol:
                break
            prev = loss
            w = w - self.learning_rate * (X.T @ r / n + self.l2 * w)
            if self.fit_intercept:
                b -= self.learning_rate * float(r.mean())
            self.n_iter_ = it
        self.coef_ = w
        self.intercept_ = b

    # --------------------------------------------------------------- prediction
    def predict(self, X: ArrayLike) -> FloatArray:
        """Return ``X w + b``."""
        self._check_fitted()
        Xa = as_2d_float(X)
        if Xa.shape[1] != self.coef_.shape[0]:
            raise ValueError(f"expected {self.coef_.shape[0]} features, got {Xa.shape[1]}")
        return Xa @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Coefficient of determination R^2 = 1 - SS_res / SS_tot."""
        return r2_score(as_1d_float(y), self.predict(X))

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("call fit() before predict()")
