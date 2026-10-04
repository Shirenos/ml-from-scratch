"""Decision boundary of a small MLP on the two-moons dataset."""

from __future__ import annotations

import matplotlib.pyplot as plt

from _common import output_dir
from ml_from_scratch import MLP, make_moons, train_test_split
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
    X, y = make_moons(500, noise=0.2, random_state=3)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=3)
    mlp = MLP((16, 16), learning_rate=0.02, n_epochs=400, batch_size=64, random_state=0)
    mlp.fit(X_tr, y_tr)
    print(f"train acc {mlp.score(X_tr, y_tr):.3f}  test acc {mlp.score(X_te, y_te):.3f}")

    xx, yy, grid = decision_grid(X)
    p = mlp.predict_proba(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.grid(False)
    cf = ax.contourf(xx, yy, p, levels=24, cmap=UNIVERSE_CMAP, alpha=0.85, vmin=0, vmax=1)
    ax.contour(xx, yy, p, levels=[0.5], colors="white", linewidths=1.6)
    ax.scatter(
        X_tr[y_tr == 0][:, 0],
        X_tr[y_tr == 0][:, 1],
        s=26,
        color=CYAN,
        edgecolor="#0a0a18",
        lw=0.5,
        label="class 0",
    )
    ax.scatter(
        X_tr[y_tr == 1][:, 0],
        X_tr[y_tr == 1][:, 1],
        s=26,
        color=PINK,
        edgecolor="#0a0a18",
        lw=0.5,
        label="class 1",
    )
    ax.scatter(
        X_te[:, 0],
        X_te[:, 1],
        s=34,
        facecolor="none",
        edgecolor="white",
        lw=0.8,
        label="test points",
    )
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("P(class 1)")
    cbar.outline.set_visible(False)
    ax.set(
        title=f"MLP (2-16-16-1, tanh) on two moons: test accuracy {mlp.score(X_te, y_te):.1%}",
        xlabel="x1",
        ylabel="x2",
    )
    ax.legend(loc="upper right")
    print(f"saved {save_figure(fig, out / 'mlp_decision_boundary.png')}")


if __name__ == "__main__":
    main()
