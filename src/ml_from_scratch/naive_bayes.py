"""Gaussian naive Bayes classifier with closed-form maximum-likelihood fitting."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._utils import FloatArray, as_2d_float, check_consistent_length, logsumexp


class GaussianNB:
    r"""Gaussian naive Bayes: features are independent normals given the class.

    Model: :math:`P(y=k) = \pi_k` and
    :math:`p(x\mid y=k) = \prod_j \mathcal N(x_j\mid\theta_{kj},\sigma_{kj}^2)`, i.e. a
    Gaussian per class with a *diagonal* covariance. Bayes' rule gives the joint
    log-likelihood

    .. math::
        \log \pi_k - \frac12\sum_j\Big[\log(2\pi\sigma_{kj}^2)
        + \frac{(x_j-\theta_{kj})^2}{\sigma_{kj}^2}\Big],

    and the posterior :math:`P(y=k\mid x)` is its softmax over ``k`` (computed with
    log-sum-exp). Fitting is closed form: the maximum-likelihood estimates are the class
    frequencies :math:`\pi_k = n_k / n`, the per-class feature means :math:`\theta_{kj}` and
    the per-class variances :math:`\sigma_{kj}^2` divided by :math:`n_k` (not
    :math:`n_k - 1`). To keep every variance positive (a constant feature inside a class,
    or a class with one sample) ``epsilon = var_smoothing * max_j Var(X_j)`` is added to
    all of them, the same rule as scikit-learn.

    Args:
        priors: Fixed class priors ``pi_k`` (in sorted class order, summing to one);
            ``None`` estimates them from the class frequencies.
        var_smoothing: Fraction of the largest feature variance added to every variance.

    Attributes:
        classes_: Sorted class labels; column ``k`` of the probabilities is ``classes_[k]``.
        class_count_: Number of training samples per class.
        class_prior_: Priors ``pi_k``.
        theta_: Per-class feature means of shape ``(n_classes, n_features)``.
        var_: Per-class feature variances (including ``epsilon_``), same shape.
        epsilon_: Absolute amount added to the variances.
    """

    def __init__(self, *, priors: ArrayLike | None = None, var_smoothing: float = 1e-9) -> None:
        if var_smoothing < 0:
            raise ValueError("var_smoothing must be non-negative")
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Estimate priors, means and variances from ``X`` of shape ``(n, d)`` and labels ``y``."""
        Xa = as_2d_float(X)
        ya = np.asarray(y)
        if ya.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {ya.shape}")
        check_consistent_length(Xa, ya)
        self.classes_: NDArray[np.generic] = np.unique(ya)
        K = len(self.classes_)

        self.epsilon_ = float(self.var_smoothing * np.max(np.var(Xa, axis=0)))
        Y = (ya[:, None] == self.classes_[None, :]).astype(np.float64)
        self.class_count_: FloatArray = Y.sum(axis=0)
        self.theta_: FloatArray = (Y.T @ Xa) / self.class_count_[:, None]
        self.var_: FloatArray = (Y.T @ Xa**2) / self.class_count_[:, None] - self.theta_**2
        # E[x^2] - E[x]^2 can dip slightly below zero in floating point
        self.var_ = np.maximum(self.var_, 0.0) + self.epsilon_
        if np.any(self.var_ <= 0.0):
            raise ValueError("a feature has zero variance; use var_smoothing > 0")

        if self.priors is None:
            self.class_prior_: FloatArray = self.class_count_ / self.class_count_.sum()
        else:
            priors = np.asarray(self.priors, dtype=np.float64)
            if priors.shape != (K,):
                raise ValueError(f"priors must have shape ({K},); got {priors.shape}")
            if np.any(priors < 0) or not np.isclose(priors.sum(), 1.0):
                raise ValueError("priors must be non-negative and sum to one")
            self.class_prior_ = priors
        return self

    def _joint_log_likelihood(self, X: ArrayLike) -> FloatArray:
        """Return ``log pi_k + log p(x | y=k)`` of shape ``(n, n_classes)``."""
        if not hasattr(self, "theta_"):
            raise RuntimeError("call fit() before predict()")
        Xa = as_2d_float(X)
        if Xa.shape[1] != self.theta_.shape[1]:
            raise ValueError(f"expected {self.theta_.shape[1]} features, got {Xa.shape[1]}")
        log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_), axis=1)
        sq = (Xa[:, None, :] - self.theta_[None, :, :]) ** 2 / self.var_[None, :, :]
        with np.errstate(divide="ignore"):
            log_prior = np.log(self.class_prior_)
        return log_prior + log_norm - 0.5 * sq.sum(axis=2)

    def predict_log_proba(self, X: ArrayLike) -> FloatArray:
        """Return ``log P(y = k | x)`` of shape ``(n, n_classes)``, normalised in log space."""
        jll = self._joint_log_likelihood(X)
        return jll - logsumexp(jll)[:, None]

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Return class probabilities of shape ``(n, n_classes)``; rows sum to one."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: ArrayLike) -> NDArray[np.generic]:
        """Return the maximum a posteriori label for every row."""
        jll = self._joint_log_likelihood(X)  # also checks that the model is fitted
        return self.classes_[np.argmax(jll, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Mean accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
