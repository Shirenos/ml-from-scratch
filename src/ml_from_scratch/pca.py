"""Principal component analysis via the singular value decomposition."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._utils import FloatArray, as_2d_float


class PCA:
    r"""Principal component analysis.

    Let :math:`X_c = X - \bar x` be the centred data (``n`` samples). PCA finds the
    orthonormal directions of maximal variance, i.e. the eigenvectors of the sample
    covariance matrix

    .. math:: C = \frac{1}{n-1} X_c^\top X_c = V \Lambda V^\top .

    Rather than forming :math:`C`, we use the thin SVD :math:`X_c = U S V^\top`:
    the rows of :math:`V^\top` are the principal axes, and the eigenvalues are
    :math:`\lambda_j = s_j^2 / (n-1)`. Projection onto the first ``k`` axes is
    :math:`Z = X_c V_k`, and the best rank-``k`` reconstruction (in the least-squares
    sense, by the Eckart-Young theorem) is :math:`\hat X = Z V_k^\top + \bar x`.

    Signs of singular vectors are arbitrary, so each component is flipped such that its
    largest-magnitude entry is positive, which makes the output deterministic.

    Args:
        n_components: Number of components to keep; ``None`` keeps ``min(n, d)``.

    Attributes:
        mean_: Per-feature mean, shape ``(d,)``.
        components_: Principal axes as rows, shape ``(k, d)``.
        explained_variance_: Variance :math:`\lambda_j` along each kept axis.
        explained_variance_ratio_: :math:`\lambda_j / \sum_i \lambda_i` (sum over *all* axes).
        singular_values_: Kept singular values :math:`s_j`.
    """

    def __init__(self, n_components: int | None = None) -> None:
        if n_components is not None and n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components

    def fit(self, X: ArrayLike) -> PCA:
        """Compute the principal axes of ``X`` of shape ``(n, d)``."""
        Xa = as_2d_float(X)
        n, d = Xa.shape
        if n < 2:
            raise ValueError("PCA needs at least 2 samples")
        k = min(n, d) if self.n_components is None else self.n_components
        if k > min(n, d):
            raise ValueError(f"n_components={k} must be <= min(n_samples, n_features)={min(n, d)}")
        self.mean_: FloatArray = Xa.mean(axis=0)
        _, s, vt = np.linalg.svd(Xa - self.mean_, full_matrices=False)
        # deterministic sign: the largest |entry| of every axis is positive
        pivot = np.abs(vt).argmax(axis=1)
        vt = vt * np.sign(vt[np.arange(vt.shape[0]), pivot])[:, None]
        variance = s**2 / (n - 1)
        self.components_: FloatArray = vt[:k]
        self.singular_values_: FloatArray = s[:k]
        self.explained_variance_: FloatArray = variance[:k]
        total = variance.sum()
        self.explained_variance_ratio_: FloatArray = (
            variance[:k] / total if total > 0 else np.zeros(k)
        )
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        """Project onto the principal axes: ``(X - mean) @ components_.T``."""
        self._check_fitted()
        Xa = as_2d_float(X)
        if Xa.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"expected {self.mean_.shape[0]} features, got {Xa.shape[1]}")
        return (Xa - self.mean_) @ self.components_.T

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        """Fit, then return the projected data."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: ArrayLike) -> FloatArray:
        """Map projected data back to the original space: ``Z @ components_ + mean``."""
        self._check_fitted()
        Za = as_2d_float(Z, "Z")
        if Za.shape[1] != self.components_.shape[0]:
            raise ValueError(f"expected {self.components_.shape[0]} components, got {Za.shape[1]}")
        return Za @ self.components_ + self.mean_

    def _check_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError("call fit() before transform()")
