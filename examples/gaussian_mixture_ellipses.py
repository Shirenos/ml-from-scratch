"""Gaussian mixture: covariance ellipses on elongated clusters, and BIC / AIC over K."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

from _common import output_dir
from ml_from_scratch import GaussianMixture, make_blobs
from ml_from_scratch.plotting import (
    CYAN,
    CYCLE,
    MUTED,
    UNIVERSE_SEQ,
    VIOLET,
    apply_universe_style,
    save_figure,
)


def draw_ellipses(ax: Axes, mean: np.ndarray, cov: np.ndarray, color: str) -> None:
    """Draw the 1, 2 and 3 sigma contours of N(mean, cov).

    With cov = V diag(lam) V^T the contour {x : (x - mean)^T cov^{-1} (x - mean) = s^2}
    is an ellipse with semi-axes s * sqrt(lam) along the eigenvectors in V.
    """
    lam, vecs = np.linalg.eigh(cov)
    angle = float(np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])))
    for s in (1.0, 2.0, 3.0):
        width, height = 2.0 * s * np.sqrt(lam[::-1])
        ax.add_patch(
            Ellipse(
                (float(mean[0]), float(mean[1])),
                float(width),
                float(height),
                angle=angle,
                fill=False,
                color=color,
                lw=1.6,
                alpha=1.0 / s,
            )
        )


def main() -> None:
    out = output_dir(__doc__ or "")
    apply_universe_style()
    centers = np.array([[0.0, 0.0], [5.0, 4.0], [-3.0, 5.0]])
    X, y = make_blobs(600, centers.tolist(), cluster_std=1.0, random_state=7)
    # A linear map keeps each blob Gaussian but makes it elongated and tilted.
    shears = [[[2.2, 0.0], [1.4, 0.5]], [[0.5, -0.9], [0.0, 1.6]], [[1.0, 0.0], [0.0, 1.0]]]
    for j, A in enumerate(np.array(shears)):
        X[y == j] = (X[y == j] - centers[j]) @ A.T + centers[j]

    gmm = GaussianMixture(3, n_init=3, random_state=0).fit(X)
    labels = gmm.predict(X)
    print(f"K=3 mean log-likelihood {gmm.lower_bound_:.4f} after {gmm.n_iter_} iterations")

    ks = np.arange(1, 8)
    fits = [GaussianMixture(int(k), n_init=3, random_state=0).fit(X) for k in ks]
    bic = [m.bic(X) for m in fits]
    aic = [m.aic(X) for m in fits]
    print(f"BIC picks K={ks[int(np.argmin(bic))]}, AIC picks K={ks[int(np.argmin(aic))]}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    pad = 1.5
    lo, hi = np.min(X, axis=0) - pad, np.max(X, axis=0) + pad
    x1 = np.linspace(lo[0], hi[0], 300)
    x2 = np.linspace(lo[1], hi[1], 300)
    g1, g2 = np.meshgrid(x1, x2)
    log_density = gmm.score_samples(np.column_stack([g1.ravel(), g2.ravel()]))
    # Clip the far tails so the colour range is spent where the data is.
    log_density = np.maximum(log_density, float(np.max(log_density)) - 10.0).reshape(g1.shape)
    ax1.contourf(g1, g2, log_density, levels=12, cmap=UNIVERSE_SEQ, alpha=0.5)
    for k in range(3):
        pts = X[labels == k]
        ax1.scatter(pts[:, 0], pts[:, 1], s=14, color=CYCLE[k], alpha=0.75, edgecolor="none")
        draw_ellipses(ax1, gmm.means_[k], gmm.covariances_[k], CYCLE[k])
    ax1.scatter(gmm.means_[:, 0], gmm.means_[:, 1], s=200, marker="*", color="white", zorder=5)
    ax1.set(
        title="GMM (K = 3): log density and 1/2/3 sigma ellipses",
        xlabel="x1",
        ylabel="x2",
        xlim=(x1[0], x1[-1]),
        ylim=(x2[0], x2[-1]),
    )

    ax2.plot(ks, bic, marker="o", color=VIOLET, mfc="white", mec=VIOLET, label="BIC")
    ax2.plot(ks, aic, marker="s", color=CYAN, mfc="white", mec=CYAN, label="AIC")
    ax2.axvline(3, color=MUTED, ls=":", lw=1)
    ax2.set(title="Model selection: BIC / AIC vs K", xlabel="K", ylabel="criterion")
    ax2.legend()
    print(f"saved {save_figure(fig, out / 'gaussian_mixture.png')}")


if __name__ == "__main__":
    main()
