"""Gaussian mixture model with full covariances, fitted by expectation-maximisation."""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from ._utils import FloatArray, IntArray, as_2d_float, check_random_state, logsumexp
from .kmeans import KMeans

Init = Literal["kmeans", "random"]

_LOG_2PI = float(np.log(2.0 * np.pi))


class _Run(NamedTuple):
    weights: FloatArray
    means: FloatArray
    covs: FloatArray
    prec: FloatArray
    history: list[float]
    converged: bool
    n_iter: int


def _precision_cholesky(covariances: FloatArray) -> FloatArray:
    """Return ``P_k = L_k^{-1}`` for ``Sigma_k = L_k L_k^T``, so ``Sigma_k^{-1} = P_k^T P_k``."""
    try:
        chol = np.linalg.cholesky(covariances)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "a covariance matrix is not positive definite; increase reg_covar"
        ) from exc
    return np.linalg.inv(chol)


def _log_gaussian(X: FloatArray, means: FloatArray, prec_chol: FloatArray) -> FloatArray:
    r"""``log N(x_i | mu_k, Sigma_k)`` for every sample and component, shape ``(n, K)``.

    With ``y = P_k (x - mu_k)`` the Mahalanobis term is :math:`\lVert y\rVert^2` and
    :math:`\log\det\Sigma_k = -2\sum_j\log (P_k)_{jj}`, so no determinant or inverse of
    :math:`\Sigma_k` is ever formed.
    """
    d = X.shape[1]
    out = np.empty((X.shape[0], means.shape[0]))
    for k in range(means.shape[0]):
        y = (X - means[k]) @ prec_chol[k].T
        log_det_prec = np.log(np.diag(prec_chol[k])).sum()
        out[:, k] = -0.5 * (d * _LOG_2PI + (y * y).sum(axis=1)) + log_det_prec
    return out


class GaussianMixture:
    r"""Gaussian mixture model fitted by expectation-maximisation (EM).

    The density is :math:`p(x) = \sum_k \pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k)`. EM maximises
    the mean log-likelihood :math:`\ell = \frac1n\sum_i\log p(x_i)` by alternating

    1. **E step**: responsibilities
       :math:`r_{ik} = \pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k) / p(x_i)`, computed in log space
       with a stable log-sum-exp;
    2. **M step**: with :math:`N_k=\sum_i r_{ik}`,
       :math:`\pi_k=N_k/n`, :math:`\mu_k=\frac1{N_k}\sum_i r_{ik}x_i`,
       :math:`\Sigma_k=\frac1{N_k}\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top`.

    For any responsibilities, Jensen's inequality gives the lower bound
    :math:`\ell \ge \frac1n\sum_{i,k} r_{ik}\log(\pi_k\mathcal N_{ik}/r_{ik})`. The E step
    makes the bound tight and the M step maximises it, so :math:`\ell` never decreases.

    ``reg_covar`` is added to the diagonal of every :math:`\Sigma_k` to keep it positive
    definite (a component collapsing onto one point would otherwise send :math:`\ell` to
    infinity). With ``reg_covar = 0`` the M step is exact and the monotonicity holds to
    round-off. The best of ``n_init`` runs (highest :math:`\ell`) is kept.

    Args:
        n_components: Number of mixture components ``K``.
        init: ``"kmeans"`` (hard k-means labels as initial responsibilities) or
            ``"random"`` (random responsibilities).
        n_init: Number of independent restarts.
        max_iter: Maximum EM iterations per restart.
        tol: Stop when the mean log-likelihood improves by at most ``tol``.
        reg_covar: Non-negative value added to the diagonal of every covariance.
        random_state: Seed (or generator).

    Attributes:
        weights_: Mixing weights :math:`\pi_k`, shape ``(K,)``.
        means_: Component means, shape ``(K, d)``.
        covariances_: Component covariances, shape ``(K, d, d)``.
        precisions_cholesky_: :math:`L_k^{-1}` with :math:`\Sigma_k = L_kL_k^\top`.
        log_likelihood_history_: Mean log-likelihood after every M step of the best run
            (the first entry follows the initial M step on the starting responsibilities).
        lower_bound_: Final mean log-likelihood of the best run.
        converged_: Whether the best run reached ``tol`` before ``max_iter``.
        n_iter_: EM iterations used by the best run.
    """

    def __init__(
        self,
        n_components: int = 1,
        *,
        init: Init = "kmeans",
        n_init: int = 1,
        max_iter: int = 100,
        tol: float = 1e-6,
        reg_covar: float = 1e-6,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if n_components < 1 or n_init < 1 or max_iter < 1:
            raise ValueError("n_components, n_init and max_iter must be >= 1")
        if init not in ("kmeans", "random"):
            raise ValueError(f"unknown init {init!r}")
        if reg_covar < 0.0:
            raise ValueError("reg_covar must be >= 0")
        self.n_components = n_components
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    # ------------------------------------------------------------------ EM steps
    def _m_step(self, X: FloatArray, resp: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        n, d = X.shape
        nk = resp.sum(axis=0) + 10.0 * np.finfo(np.float64).eps
        means = (resp.T @ X) / nk[:, None]
        covs = np.empty((self.n_components, d, d))
        for k in range(self.n_components):
            diff = X - means[k]
            covs[k] = (resp[:, k, None] * diff).T @ diff / nk[k]
            covs[k].flat[:: d + 1] += self.reg_covar
        return nk / n, means, covs

    @staticmethod
    def _weighted_log_prob(
        X: FloatArray, weights: FloatArray, means: FloatArray, prec_chol: FloatArray
    ) -> FloatArray:
        return _log_gaussian(X, means, prec_chol) + np.log(weights)

    def _initial_resp(self, X: FloatArray, rng: np.random.Generator) -> FloatArray:
        n, k = X.shape[0], self.n_components
        if self.init == "random":
            resp = rng.random((n, k))
            return resp / resp.sum(axis=1, keepdims=True)
        labels = KMeans(k, n_init=1, random_state=rng).fit_predict(X)
        resp = np.zeros((n, k))
        resp[np.arange(n), labels] = 1.0
        return resp

    def _run(self, X: FloatArray, rng: np.random.Generator) -> _Run:
        weights, means, covs = self._m_step(X, self._initial_resp(X, rng))
        history: list[float] = []
        converged = False
        n_iter = 0
        while True:
            prec = _precision_cholesky(covs)
            wlp = self._weighted_log_prob(X, weights, means, prec)
            log_norm = logsumexp(wlp)
            history.append(float(log_norm.mean()))
            if len(history) > 1 and abs(history[-1] - history[-2]) <= self.tol:
                converged = True
                break
            if n_iter == self.max_iter:
                break
            resp = np.exp(wlp - log_norm[:, None])  # E step
            weights, means, covs = self._m_step(X, resp)  # M step
            n_iter += 1
        return _Run(weights, means, covs, prec, history, converged, n_iter)

    # ------------------------------------------------------------------ public API
    def fit(self, X: ArrayLike) -> GaussianMixture:
        """Fit the mixture to ``X`` of shape ``(n, d)``."""
        Xa = as_2d_float(X)
        if Xa.shape[0] < self.n_components:
            raise ValueError(f"n_samples={Xa.shape[0]} < n_components={self.n_components}")
        rng = check_random_state(self.random_state)
        best: _Run | None = None
        for _ in range(self.n_init):
            run = self._run(Xa, rng)
            if best is None or run.history[-1] > best.history[-1]:
                best = run
        assert best is not None
        self.weights_ = best.weights
        self.means_ = best.means
        self.covariances_ = best.covs
        self.precisions_cholesky_ = best.prec
        self.log_likelihood_history_ = best.history
        self.lower_bound_ = best.history[-1]
        self.converged_ = best.converged
        self.n_iter_ = best.n_iter
        return self

    def _check(self, X: ArrayLike) -> FloatArray:
        if not hasattr(self, "means_"):
            raise RuntimeError("call fit() before using the model")
        Xa = as_2d_float(X)
        if Xa.shape[1] != self.means_.shape[1]:
            raise ValueError(f"expected {self.means_.shape[1]} features, got {Xa.shape[1]}")
        return Xa

    def _wlp(self, X: ArrayLike) -> FloatArray:
        return self._weighted_log_prob(
            self._check(X), self.weights_, self.means_, self.precisions_cholesky_
        )

    def score_samples(self, X: ArrayLike) -> FloatArray:
        """Log-density ``log p(x)`` of every row of ``X``."""
        return logsumexp(self._wlp(X))

    def score(self, X: ArrayLike) -> float:
        """Mean log-likelihood of ``X``."""
        return float(self.score_samples(X).mean())

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Posterior responsibilities ``P(component k | x)``, shape ``(n, K)``."""
        wlp = self._wlp(X)
        return np.exp(wlp - logsumexp(wlp)[:, None])

    def predict(self, X: ArrayLike) -> IntArray:
        """Most probable component for every row of ``X``."""
        return self._wlp(X).argmax(axis=1).astype(np.intp)

    def fit_predict(self, X: ArrayLike) -> IntArray:
        """Fit and return the component labels of the training data."""
        return self.fit(X).predict(X)

    def n_parameters(self) -> int:
        """Number of free parameters: ``K-1`` weights, ``K d`` means, ``K d(d+1)/2`` covariances."""
        k, d = self.means_.shape
        return (k - 1) + k * d + k * d * (d + 1) // 2

    def bic(self, X: ArrayLike) -> float:
        """Bayesian information criterion ``-2 log L + p log n`` (lower is better)."""
        n = self._check(X).shape[0]
        return -2.0 * self.score(X) * n + self.n_parameters() * float(np.log(n))

    def aic(self, X: ArrayLike) -> float:
        """Akaike information criterion ``-2 log L + 2 p`` (lower is better)."""
        n = self._check(X).shape[0]
        return -2.0 * self.score(X) * n + 2.0 * self.n_parameters()

    def sample(
        self, n_samples: int, random_state: int | np.random.Generator | None = None
    ) -> tuple[FloatArray, IntArray]:
        """Draw ``n_samples`` points; returns ``(X, component_labels)``."""
        if not hasattr(self, "means_"):
            raise RuntimeError("call fit() before sample()")
        rng = check_random_state(random_state)
        labels = rng.choice(self.n_components, size=n_samples, p=self.weights_).astype(np.intp)
        chol = np.linalg.cholesky(self.covariances_)
        z = rng.standard_normal((n_samples, self.means_.shape[1]))
        X = self.means_[labels] + np.einsum("nij,nj->ni", chol[labels], z)
        return X, labels
