"""Gaussian mixture: the four covariance types fitted to the same elongated clusters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import output_dir
from gaussian_mixture_ellipses import draw_ellipses, sheared_blobs
from ml_from_scratch import GaussianMixture
from ml_from_scratch.gaussian_mixture import CovarianceType
from ml_from_scratch.plotting import CYCLE, apply_universe_style, save_figure

# Each type restricts what one component's covariance may look like.
TYPES: dict[CovarianceType, str] = {
    "full": "own full matrix",
    "tied": "one shared full matrix",
    "diag": "own axis-aligned matrix",
    "spherical": "own multiple of I",
}


def full_covariances(gmm: GaussianMixture) -> np.ndarray:
    """Expand ``covariances_`` (sklearn's per-type layout) to ``(K, d, d)`` matrices."""
    cov = np.asarray(gmm.covariances_)
    k, d = gmm.means_.shape
    if gmm.covariance_type == "full":
        return cov
    if gmm.covariance_type == "tied":
        return np.broadcast_to(cov, (k, d, d))
    if gmm.covariance_type == "diag":
        return np.stack([np.diag(c) for c in cov])
    return cov[:, None, None] * np.eye(d)


def main() -> None:
    out = output_dir(__doc__ or "")
    apply_universe_style()
    X = sheared_blobs()

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), sharex=True, sharey=True)
    pad = 1.5
    lo, hi = np.min(X, axis=0) - pad, np.max(X, axis=0) + pad
    for ax, (cov_type, meaning) in zip(axes.ravel(), TYPES.items(), strict=True):
        gmm = GaussianMixture(3, covariance_type=cov_type, n_init=3, random_state=0).fit(X)
        labels = gmm.predict(X)
        covs = full_covariances(gmm)
        bic = gmm.bic(X)
        print(f"{cov_type:>9}: {gmm.n_parameters()} parameters, BIC {bic:.1f}")
        for k in range(3):
            pts = X[labels == k]
            ax.scatter(pts[:, 0], pts[:, 1], s=10, color=CYCLE[k], alpha=0.6, edgecolor="none")
            draw_ellipses(ax, gmm.means_[k], covs[k], CYCLE[k])
        ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], s=160, marker="*", color="white", zorder=5)
        ax.set_title(
            f"{cov_type}: {meaning}\n{gmm.n_parameters()} params, BIC = {bic:.0f}", fontsize=10
        )
    for ax in axes[-1]:
        ax.set_xlabel("x1")
    for ax in axes[:, 0]:
        ax.set_ylabel("x2")
    axes[0, 0].set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    fig.suptitle("GMM covariance types (K = 3, lower BIC is better)")
    print(f"saved {save_figure(fig, out / 'gmm_covariance_types.png')}")


if __name__ == "__main__":
    main()
