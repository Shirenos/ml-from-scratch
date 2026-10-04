"""Tiny synthetic datasets (NumPy only) used by the tests and examples."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._utils import FloatArray, IntArray, check_random_state


def make_moons(
    n_samples: int = 200,
    *,
    noise: float = 0.1,
    random_state: int | np.random.Generator | None = None,
) -> tuple[FloatArray, IntArray]:
    """Two interleaving half circles ("two moons"), with Gaussian noise of std ``noise``.

    Returns ``X`` of shape ``(n_samples, 2)`` and labels ``y`` in ``{0, 1}``.
    """
    rng = check_random_state(random_state)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    t_out = np.linspace(0.0, np.pi, n_out)
    t_in = np.linspace(0.0, np.pi, n_in)
    upper = np.column_stack([np.cos(t_out), np.sin(t_out)])
    lower = np.column_stack([1.0 - np.cos(t_in), 0.5 - np.sin(t_in)])
    X = np.vstack([upper, lower]) + rng.normal(0.0, noise, size=(n_samples, 2))
    y = np.concatenate([np.zeros(n_out), np.ones(n_in)]).astype(np.intp)
    return X, y


def make_blobs(
    n_samples: int = 300,
    centers: ArrayLike | int = 3,
    *,
    cluster_std: float = 0.6,
    n_features: int = 2,
    random_state: int | np.random.Generator | None = None,
) -> tuple[FloatArray, IntArray]:
    """Isotropic Gaussian blobs.

    ``centers`` is either an array of shape ``(k, n_features)`` or an integer ``k``
    (centres are then drawn uniformly from ``[-6, 6]^n_features``).
    """
    rng = check_random_state(random_state)
    if isinstance(centers, int):
        c = rng.uniform(-6.0, 6.0, size=(centers, n_features))
    else:
        c = np.asarray(centers, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError("centers must have shape (k, n_features)")
    k = c.shape[0]
    y = np.arange(n_samples) % k
    X = c[y] + rng.normal(0.0, cluster_std, size=(n_samples, c.shape[1]))
    return X, y.astype(np.intp)


def train_test_split(
    X: FloatArray,
    y: IntArray,
    *,
    test_size: float = 0.25,
    random_state: int | np.random.Generator | None = None,
) -> tuple[FloatArray, FloatArray, IntArray, IntArray]:
    """Random split into ``X_train, X_test, y_train, y_test``."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    rng = check_random_state(random_state)
    perm = rng.permutation(len(X))
    n_test = max(1, round(len(X) * test_size))
    test, train = perm[:n_test], perm[n_test:]
    return X[train], X[test], y[train], y[test]
