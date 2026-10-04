"""k-means clustering (Lloyd's algorithm with k-means++ seeding)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._utils import FloatArray, IntArray, as_2d_float, check_random_state

Init = Literal["k-means++", "random"]


def _sq_dists(X: FloatArray, centers: FloatArray) -> FloatArray:
    """Squared Euclidean distances, shape ``(n, k)``, via ``|x|^2 - 2 x.c + |c|^2``."""
    d = (X * X).sum(axis=1)[:, None] - 2.0 * X @ centers.T + (centers * centers).sum(axis=1)
    return np.maximum(d, 0.0)  # guard against tiny negative values from round-off


class KMeans:
    r"""k-means clustering.

    Minimises the within-cluster sum of squares (inertia)

    .. math:: J = \sum_{i=1}^{n} \lVert x_i - \mu_{c_i}\rVert^2

    over assignments :math:`c_i` and centroids :math:`\mu_k` by alternating two steps
    (Lloyd's algorithm); each step cannot increase :math:`J`:

    1. **Assign**: :math:`c_i = \arg\min_k \lVert x_i - \mu_k\rVert^2`
    2. **Update**: :math:`\mu_k = \frac{1}{|C_k|}\sum_{i \in C_k} x_i` (the mean minimises
       the squared distance to the points of the cluster)

    The problem is non-convex, so the result depends on the start. **k-means++**
    seeding picks the first centre uniformly and every next centre with probability
    proportional to :math:`D(x)^2`, the squared distance to the nearest chosen centre;
    this gives an :math:`O(\log k)`-competitive solution in expectation. The best of
    ``n_init`` runs (lowest inertia) is kept. An empty cluster is re-seeded with the
    point that is farthest from its centroid.

    Args:
        n_clusters: Number of clusters ``k``.
        init: ``"k-means++"`` or ``"random"`` (k distinct data points).
        n_init: Number of independent restarts.
        max_iter: Maximum Lloyd iterations per restart.
        tol: Stop when the total squared centroid shift is at most ``tol``.
        random_state: Seed (or generator).

    Attributes:
        cluster_centers_: Centroids, shape ``(k, d)``.
        labels_: Cluster index of every training sample.
        inertia_: Final value of ``J``.
        n_iter_: Iterations used by the best run.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        *,
        init: Init = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-8,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if n_clusters < 1 or n_init < 1 or max_iter < 1:
            raise ValueError("n_clusters, n_init and max_iter must be >= 1")
        if init not in ("k-means++", "random"):
            raise ValueError(f"unknown init {init!r}")
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_centers(self, X: FloatArray, rng: np.random.Generator) -> FloatArray:
        n = X.shape[0]
        k = self.n_clusters
        if self.init == "random":
            return X[rng.choice(n, size=k, replace=False)].copy()
        centers = np.empty((k, X.shape[1]))
        centers[0] = X[rng.integers(n)]
        closest = _sq_dists(X, centers[:1])[:, 0]
        for j in range(1, k):
            total = closest.sum()
            # if total == 0 all remaining points coincide with a centre: pick uniformly
            probs = closest / total if total > 0.0 else None
            idx = int(rng.choice(n, p=probs))
            centers[j] = X[idx]
            closest = np.minimum(closest, _sq_dists(X, centers[j : j + 1])[:, 0])
        return centers

    def _lloyd(self, X: FloatArray, centers: FloatArray) -> tuple[FloatArray, IntArray, float, int]:
        k = self.n_clusters
        n_iter = 0
        for iteration in range(1, self.max_iter + 1):
            n_iter = iteration
            d = _sq_dists(X, centers)
            labels = d.argmin(axis=1)
            fit_cost: FloatArray = d[np.arange(len(X)), labels]
            new = np.empty_like(centers)
            for j in range(k):
                members = X[labels == j]
                if len(members):
                    new[j] = members.mean(axis=0)
                else:  # re-seed an empty cluster with the worst-fitted point
                    worst = int(np.argmax(fit_cost))
                    new[j] = X[worst]
                    labels[worst] = j
                    fit_cost[worst] = 0.0
            shift = float(((new - centers) ** 2).sum())
            centers = new
            if shift <= self.tol:
                break
        d = _sq_dists(X, centers)
        labels = d.argmin(axis=1)
        inertia = float(d[np.arange(len(X)), labels].sum())
        return centers, labels.astype(np.intp), inertia, n_iter

    def fit(self, X: ArrayLike) -> KMeans:
        """Cluster ``X`` of shape ``(n, d)``."""
        Xa = as_2d_float(X)
        if Xa.shape[0] < self.n_clusters:
            raise ValueError(f"n_samples={Xa.shape[0]} < n_clusters={self.n_clusters}")
        rng = check_random_state(self.random_state)
        best: tuple[FloatArray, IntArray, float, int] | None = None
        for _ in range(self.n_init):
            run = self._lloyd(Xa, self._init_centers(Xa, rng))
            if best is None or run[2] < best[2]:
                best = run
        assert best is not None
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X: ArrayLike) -> IntArray:
        """Index of the nearest centroid for every row of ``X``."""
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("call fit() before predict()")
        Xa = as_2d_float(X)
        if Xa.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                f"expected {self.cluster_centers_.shape[1]} features, got {Xa.shape[1]}"
            )
        return _sq_dists(Xa, self.cluster_centers_).argmin(axis=1).astype(np.intp)

    def fit_predict(self, X: ArrayLike) -> IntArray:
        """Fit and return ``labels_``."""
        return self.fit(X).labels_
