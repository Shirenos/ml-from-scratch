"""Principal component analysis via the singular value decomposition."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._utils import FloatArray, as_2d_float


def _flip_singular_vectors(vt: FloatArray) -> FloatArray:
    """Flip each right singular vector so its first non-zero entry is positive.

    Singular vectors are unique only up to sign. Choosing the first non-zero
    loading as the pivot avoids ``np.sign(0) == 0``, which would otherwise
    multiply an entire component by zero (the failure mode of using a pivot
    that can be exactly 0). This matches the spirit of scikit-learn's
    ``svd_flip``: pick a non-zero reference loading and force it positive.
    """
    oriented = np.array(vt, dtype=float, copy=True)
    # argmax on a bool mask yields the index of the first True; all-False -> 0.
    nonzero = oriented != 0
    has_nonzero = nonzero.any(axis=1)
    first_nz = nonzero.argmax(axis=1)
    rows = np.arange(oriented.shape[0])
    signs = np.ones(oriented.shape[0], dtype=float)
    signs[has_nonzero] = np.sign(oriented[rows[has_nonzero], first_nz[has_nonzero]])
    oriented *= signs[:, None]
    return oriented


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
    first non-zero entry is positive, which makes the output deterministic. Using the
    first non-zero entry (rather than ``np.sign`` of a possibly zero pivot) avoids
    wiping out a whole component when the reference loading is exactly 0.

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
        # Deterministic sign (sklearn svd_flip style): first non-zero entry positive.
        # np.sign(0) == 0 would zero out a whole row if the pivot loading were 0.
        vt = _flip_singular_vectors(vt)
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
