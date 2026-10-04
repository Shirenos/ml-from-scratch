"""Logistic regression: linear decision boundary and probability map."""

from __future__ import annotations

import matplotlib.pyplot as plt

from _common import output_dir
from ml_from_scratch import LogisticRegression, make_blobs
from ml_from_scratch.plotting import (
    CYAN,
    PINK,
    UNIVERSE_CMAP,
    apply_universe_style,
    decision_grid,
    save_figure,
)


def main() -> None:
    out = output_dir(__doc__ or "")
    apply_universe_style()
    X, y = make_blobs(300, [[-1.4, -1.0], [1.4, 1.0]], cluster_std=1.0, random_state=5)
    model = LogisticRegression(learning_rate=0.5, n_iter=500, l2=1e-3).fit(X, y)
    print(f"accuracy {model.score(X, y):.3f}, w={model.coef_}, b={model.intercept_:.3f}")

    xx, yy, grid = decision_grid(X)
    p = model.predict_proba(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.grid(False)
    ax.contourf(xx, yy, p, levels=24, cmap=UNIVERSE_CMAP, alpha=0.85, vmin=0, vmax=1)
    ax.contour(xx, yy, p, levels=[0.5], colors="white", linewidths=1.6)
    ax.scatter(
        X[y == 0][:, 0],
        X[y == 0][:, 1],
        s=26,
        color=CYAN,
        edgecolor="#0a0a18",
        lw=0.5,
        label="class 0",
    )
    ax.scatter(
        X[y == 1][:, 0],
        X[y == 1][:, 1],
        s=26,
        color=PINK,
        edgecolor="#0a0a18",
        lw=0.5,
        label="class 1",
    )
    ax.set(title=f"Logistic regression: accuracy {model.score(X, y):.1%}", xlabel="x1", ylabel="x2")
    ax.legend(loc="upper left")
    print(f"saved {save_figure(fig, out / 'logistic_regression.png')}")


if __name__ == "__main__":
    main()
