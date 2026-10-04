"""k-means: clusters with centroids, and the inertia ("elbow") curve over k."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import output_dir
from ml_from_scratch import KMeans, make_blobs
from ml_from_scratch.plotting import CYCLE, MUTED, VIOLET, apply_universe_style, save_figure


def main() -> None:
    out = output_dir(__doc__ or "")
    apply_universe_style()
    true_centers = [[-5, -3], [0, 5], [5, -2], [6, 6]]
    X, _ = make_blobs(480, true_centers, cluster_std=1.1, random_state=11)
    model = KMeans(4, random_state=0).fit(X)
    print(f"k=4 inertia {model.inertia_:.2f} after {model.n_iter_} iterations")

    ks = np.arange(1, 9)
    inertias = [KMeans(int(k), random_state=0).fit(X).inertia_ for k in ks]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    for j in range(4):
        pts = X[model.labels_ == j]
        ax1.scatter(pts[:, 0], pts[:, 1], s=24, color=CYCLE[j], alpha=0.8, edgecolor="none")
    ax1.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        s=240,
        marker="*",
        color="white",
        edgecolor="#0a0a18",
        lw=1.2,
        zorder=5,
        label="centroids",
    )
    ax1.set(title="k-means (k = 4, k-means++ init)", xlabel="x1", ylabel="x2")
    ax1.legend(loc="upper left")

    ax2.plot(ks, inertias, marker="o", color=VIOLET, mfc="white", mec=VIOLET)
    ax2.axvline(4, color=MUTED, ls=":", lw=1)
    ax2.set(title="Elbow: inertia vs k", xlabel="k", ylabel="inertia  J")
    print(f"saved {save_figure(fig, out / 'kmeans_clusters.png')}")


if __name__ == "__main__":
    main()
