"""Softmax regression: three-class decision regions and prediction confidence."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array

from _common import output_dir
from ml_from_scratch import SoftmaxRegression, make_blobs
from ml_from_scratch.plotting import (
    BG,
    CYAN,
    PINK,
    VIOLET,
    apply_universe_style,
    decision_grid,
    save_figure,
)


def main() -> None:
    out = output_dir(__doc__ or "")
    apply_universe_style()
    X, y = make_blobs(450, [[-2.2, -1.2], [2.2, -1.2], [0.0, 2.4]], cluster_std=1.1, random_state=7)
    model = SoftmaxRegression(learning_rate=0.5, n_iter=2000, l2=1e-3).fit(X, y)
    print(f"accuracy {model.score(X, y):.3f}, final loss {model.loss_history_[-1]:.4f}")

    colors = [CYAN, PINK, VIOLET]
    xx, yy, grid = decision_grid(X)
    proba = model.predict_proba(grid)
    # colour every pixel by its predicted class, more opaque where the model is confident
    rgb = to_rgba_array(colors)[proba.argmax(axis=1), :3]
    confidence = (proba.max(axis=1) - 1 / 3) / (2 / 3)
    image = np.column_stack([rgb, 0.1 + 0.5 * confidence]).reshape(*xx.shape, 4)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.grid(False)
    extent = (xx.min(), xx.max(), yy.min(), yy.max())
    ax.imshow(image, origin="lower", extent=extent, aspect="auto", interpolation="bilinear")
    # boundaries are where a class probability equals the largest of the others
    for k in range(len(colors)):
        margin = proba[:, k] - np.delete(proba, k, axis=1).max(axis=1)
        ax.contour(xx, yy, margin.reshape(xx.shape), levels=[0.0], colors="white", linewidths=1.4)
    for k, color in enumerate(colors):
        pts = X[y == k]
        ax.scatter(
            pts[:, 0], pts[:, 1], s=24, color=color, edgecolor=BG, lw=0.5, label=f"class {k}"
        )
    ax.set(
        title=f"Softmax regression: accuracy {model.score(X, y):.1%}",
        xlabel="x1",
        ylabel="x2",
        xlim=(xx.min(), xx.max()),
        ylim=(yy.min(), yy.max()),
    )
    ax.legend(loc="upper left")
    print(f"saved {save_figure(fig, out / 'softmax_regression.png')}")


if __name__ == "__main__":
    main()
