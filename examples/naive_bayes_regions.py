"""Gaussian naive Bayes: quadratic decision regions and per-class axis-aligned Gaussians."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba_array
from matplotlib.patches import Ellipse

from _common import output_dir
from ml_from_scratch import GaussianNB, make_blobs
from ml_from_scratch.plotting import (
    BG,
    CYAN,
    PINK,
    VIOLET,
    apply_universe_style,
    decision_grid,
    save_figure,
)


def draw_axis_aligned(ax: Axes, mean: np.ndarray, var: np.ndarray, color: str) -> None:
    """Draw the 1 and 2 sigma contours of N(mean, diag(var)).

    With a diagonal covariance the contour sum_j (x_j - mean_j)^2 / var_j = s^2 is an
    ellipse whose semi-axes s * sqrt(var_j) are parallel to the coordinate axes; this is
    exactly the shape the naive (conditional independence) assumption allows.
    """
    for s in (1.0, 2.0):
        width, height = 2.0 * s * np.sqrt(var)
        ax.add_patch(
            Ellipse(
                (float(mean[0]), float(mean[1])),
                float(width),
                float(height),
                fill=False,
                color=color,
                lw=1.6,
                alpha=1.0 / s,
            )
        )


def main() -> None:
    out = output_dir(__doc__ or "")
    apply_universe_style()
    centers = np.array([[-2.5, -1.0], [2.5, -1.0], [0.0, 2.8]])
    X, y = make_blobs(450, centers.tolist(), cluster_std=1.0, random_state=3)
    # Per-class axis scales keep the features independent within each class but give the
    # classes different variances, so the Bayes boundaries become quadratic curves.
    scales = np.array([[1.6, 0.5], [0.6, 1.5], [1.0, 0.8]])
    for k, scale in enumerate(scales):
        X[y == k] = (X[y == k] - centers[k]) * scale + centers[k]

    model = GaussianNB().fit(X, y)
    print(f"accuracy {model.score(X, y):.3f}, priors {np.round(model.class_prior_, 3)}")

    colors = [CYAN, PINK, VIOLET]
    xx, yy, grid = decision_grid(X)
    proba = model.predict_proba(grid)
    # colour every pixel by its predicted class, more opaque where the model is confident
    rgb = to_rgba_array(colors)[proba.argmax(axis=1), :3]
    confidence = (proba.max(axis=1) - 1 / 3) / (2 / 3)
    image = np.column_stack([rgb, 0.08 + 0.4 * confidence]).reshape(*xx.shape, 4)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.grid(False)
    extent = (xx.min(), xx.max(), yy.min(), yy.max())
    ax.imshow(image, origin="lower", extent=extent, aspect="auto", interpolation="bilinear")
    for k in range(len(colors)):
        margin = proba[:, k] - np.delete(proba, k, axis=1).max(axis=1)
        ax.contour(xx, yy, margin.reshape(xx.shape), levels=[0.0], colors="white", linewidths=1.4)
    for k, color in enumerate(colors):
        pts = X[y == k]
        ax.scatter(
            pts[:, 0], pts[:, 1], s=22, color=color, edgecolor=BG, lw=0.5, label=f"class {k}"
        )
        draw_axis_aligned(ax, model.theta_[k], model.var_[k], color)
        ax.plot(*model.theta_[k], marker="X", ms=11, color="white", mec=BG, mew=1.2)
    ax.set(
        title=f"Gaussian naive Bayes: accuracy {model.score(X, y):.1%}",
        xlabel="x1",
        ylabel="x2",
        xlim=(xx.min(), xx.max()),
        ylim=(yy.min(), yy.max()),
    )
    ax.legend(loc="upper left")
    print(f"saved {save_figure(fig, out / 'naive_bayes.png')}")


if __name__ == "__main__":
    main()
