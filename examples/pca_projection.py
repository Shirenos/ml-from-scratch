"""PCA: project 6-D clustered data onto its first two principal components."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import output_dir
from ml_from_scratch import PCA, make_blobs
from ml_from_scratch.plotting import CYAN, CYCLE, MUTED, apply_universe_style, save_figure


def main() -> None:
    out = output_dir(__doc__ or "")
    apply_universe_style()
    rng = np.random.default_rng(2)
    # three clusters living in a 2-D subspace of R^6, plus a little isotropic noise
    Z, y = make_blobs(450, [[-4, 0], [3, 3], [3, -3]], cluster_std=0.9, random_state=2)
    basis = np.linalg.qr(rng.normal(size=(6, 2)))[0].T * np.array([[2.0], [1.2]])
    X = Z @ basis + rng.normal(0, 0.15, size=(450, 6))

    pca = PCA(2).fit(X)
    proj = pca.transform(X)
    full = PCA().fit(X)
    ratio = full.explained_variance_ratio_
    print("explained variance ratio:", np.round(ratio, 4))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    for c in range(3):
        ax1.scatter(
            proj[y == c][:, 0],
            proj[y == c][:, 1],
            s=24,
            color=CYCLE[c],
            alpha=0.8,
            edgecolor="none",
        )
    ax1.set(
        title=f"PCA projection of 6-D data ({ratio[:2].sum():.1%} of variance kept)",
        xlabel=f"PC1 ({ratio[0]:.1%})",
        ylabel=f"PC2 ({ratio[1]:.1%})",
    )
    ax1.set_aspect("equal", adjustable="datalim")

    idx = np.arange(1, len(ratio) + 1)
    ax2.bar(idx, ratio, color=CYAN, alpha=0.85, label="per component")
    ax2.plot(idx, np.cumsum(ratio), marker="o", color=CYCLE[2], mfc="white", label="cumulative")
    ax2.axhline(1.0, color=MUTED, ls=":", lw=1)
    ax2.set(title="Explained variance", xlabel="component", ylabel="ratio")
    ax2.legend(loc="center right")
    print(f"saved {save_figure(fig, out / 'pca_projection.png')}")


if __name__ == "__main__":
    main()
