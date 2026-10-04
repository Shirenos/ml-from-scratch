"""Shared helpers: input validation and numerically stable math primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


def as_2d_float(X: ArrayLike, name: str = "X") -> FloatArray:
    """Convert ``X`` to a finite float64 array of shape ``(n_samples, n_features)``."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features); got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinity")
    return arr


def as_1d_float(y: ArrayLike, name: str = "y") -> FloatArray:
    """Convert ``y`` to a finite float64 vector of shape ``(n_samples,)``."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinity")
    return arr


def check_consistent_length(X: FloatArray, y: Any) -> None:
    """Raise if ``X`` and ``y`` have a different number of samples."""
    if X.shape[0] != len(y):
        raise ValueError(f"X has {X.shape[0]} samples but y has {len(y)}")


def check_random_state(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` from a seed, a generator or ``None``."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def sigmoid(z: FloatArray) -> FloatArray:
    r"""Logistic function :math:`\sigma(z) = 1 / (1 + e^{-z})`, stable for large ``|z|``."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def softplus(z: FloatArray) -> FloatArray:
    """Compute ``log(1 + exp(z))`` without overflow."""
    return np.logaddexp(0.0, z)


def softmax(z: FloatArray) -> FloatArray:
    """Row-wise softmax ``exp(z_k) / sum_j exp(z_j)`` with the max subtracted for stability."""
    shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def logsumexp(z: FloatArray) -> FloatArray:
    """Row-wise ``log(sum_j exp(z_j))``, stable."""
    m = z.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True))).ravel()


def r2_score(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot`` (averaged over outputs)."""
    y_true = y_true.reshape(len(y_true), -1)
    y_pred = y_pred.reshape(len(y_pred), -1)
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    safe = np.where(ss_tot == 0.0, 1.0, ss_tot)
    r2 = np.where(ss_tot == 0.0, np.where(ss_res == 0.0, 1.0, 0.0), 1.0 - ss_res / safe)
    return float(r2.mean())
