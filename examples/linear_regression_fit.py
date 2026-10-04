"""Linear regression: closed form vs gradient descent on noisy 1-D data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import output_dir
from ml_from_scratch import LinearRegression
from ml_from_scratch.plotting import CYAN, MUTED, PINK, VIOLET, apply_universe_style, save_figure


def main() -> None:
    out = output_dir(__doc__ or "")
    apply_universe_style()
    rng = np.random.default_rng(7)
    x = rng.uniform(-3, 3, 80)
    y = 1.8 * x + 0.7 + rng.normal(0, 1.0, x.size)
    X = x[:, None]

    closed = LinearRegression("closed_form").fit(X, y)
    descent = LinearRegression("gradient_descent", learning_rate=0.05, n_iter=150).fit(X, y)
    print(f"closed form : w={closed.coef_[0]:.4f} b={closed.intercept_:.4f}")
    print(f"grad descent: w={descent.coef_[0]:.4f} b={descent.intercept_:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = np.linspace(-3.3, 3.3, 100)[:, None]
    ax1.scatter(x, y, s=28, color=VIOLET, alpha=0.85, edgecolor="none", label="data")
    ax1.plot(grid, closed.predict(grid), color=CYAN, label="closed form")
    ax1.plot(grid, descent.predict(grid), color=PINK, ls="--", label="gradient descent")
    ax1.set(title="Linear regression fit", xlabel="x", ylabel="y")
    ax1.legend()

    losses = np.array(descent.loss_history_)
    optimum = 0.5 * np.mean((closed.predict(X) - y) ** 2)
    ax2.plot(losses - optimum + 1e-12, color=PINK)
    ax2.set_yscale("log")
    ax2.set(
        title="Gradient descent converges to the closed-form optimum",
        xlabel="iteration",
        ylabel="J(w, b) - J*",
    )
    ax2.axhline(1e-12, color=MUTED, lw=0.8, ls=":")
    path = save_figure(fig, out / "linear_regression.png")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
