"""Training loss curves of linear regression (GD), logistic regression and the MLP."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import output_dir
from ml_from_scratch import MLP, LinearRegression, LogisticRegression, make_blobs, make_moons
from ml_from_scratch.plotting import CYAN, PINK, VIOLET, apply_universe_style, save_figure


def main() -> None:
    out = output_dir(__doc__ or "")
    apply_universe_style()
    rng = np.random.default_rng(0)

    X = rng.normal(size=(300, 4))
    y = X @ np.array([1.5, -2.0, 0.0, 0.7]) + 1.0 + rng.normal(0, 0.3, 300)
    lin = LinearRegression("gradient_descent", learning_rate=0.1, n_iter=100).fit(X, y)

    Xb, yb = make_blobs(400, [[-1.2, -1.0], [1.2, 1.0]], cluster_std=1.0, random_state=1)
    log = LogisticRegression(learning_rate=0.3, n_iter=150).fit(Xb, yb)

    Xm, ym = make_moons(400, noise=0.15, random_state=0)
    mlp = MLP((16, 16), learning_rate=0.01, n_epochs=300, batch_size=64, random_state=0)
    mlp.fit(Xm, ym)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    panels = [
        ("Linear regression (MSE / 2)", lin.loss_history_, VIOLET, "iteration"),
        ("Logistic regression (cross-entropy)", log.loss_history_, CYAN, "iteration"),
        ("MLP on two moons (cross-entropy)", mlp.loss_history_, PINK, "epoch"),
    ]
    for ax, (title, history, color, xlabel) in zip(axes, panels, strict=True):
        ax.plot(history, color=color)
        ax.fill_between(range(len(history)), history, min(history), color=color, alpha=0.12)
        ax.set(title=title, xlabel=xlabel, ylabel="loss")
    axes[0].set_yscale("log")
    print(
        f"final losses: {lin.loss_history_[-1]:.4f} {log.loss_history_[-1]:.4f} "
        f"{mlp.loss_history_[-1]:.4f}"
    )
    fig.tight_layout()
    print(f"saved {save_figure(fig, out / 'loss_curves.png')}")


if __name__ == "__main__":
    main()
