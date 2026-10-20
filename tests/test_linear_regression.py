import warnings

import numpy as np
import pytest

from ml_from_scratch import DivergenceWarning, LinearRegression
from ml_from_scratch._utils import feature_scaling


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + 3.0 + rng.normal(0, 0.1, 200)
    return X, y


def test_exact_line() -> None:
    x = np.arange(10, dtype=float)[:, None]
    model = LinearRegression().fit(x, 2.0 * x[:, 0] + 1.0)
    assert model.coef_ == pytest.approx([2.0])
    assert model.intercept_ == pytest.approx(1.0)
    assert model.score(x, 2.0 * x[:, 0] + 1.0) == pytest.approx(1.0)


def test_normal_equations(data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = data
    A = np.column_stack([X, np.ones(len(X))])
    expected = np.linalg.solve(A.T @ A, A.T @ y)  # (X'X)^-1 X'y
    model = LinearRegression().fit(X, y)
    assert np.allclose(np.append(model.coef_, model.intercept_), expected)


def test_gradient_descent_matches_closed_form(data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = data
    cf = LinearRegression().fit(X, y)
    gd = LinearRegression("gradient_descent", learning_rate=0.1, n_iter=5000).fit(X, y)
    assert np.allclose(gd.coef_, cf.coef_, atol=1e-5)
    assert gd.intercept_ == pytest.approx(cf.intercept_, abs=1e-5)


def test_gradient_descent_loss_decreases(data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = data
    gd = LinearRegression("gradient_descent", n_iter=200).fit(X, y)
    losses = np.array(gd.loss_history_)
    assert np.all(np.diff(losses) <= 1e-12)
    assert losses[-1] < 0.01 * losses[0]


def test_ridge_closed_form_and_gd_agree(data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = data
    cf = LinearRegression(l2=0.5).fit(X, y)
    gd = LinearRegression("gradient_descent", l2=0.5, n_iter=5000).fit(X, y)
    assert np.allclose(gd.coef_, cf.coef_, atol=1e-5)
    # penalty shrinks the weights
    assert np.linalg.norm(cf.coef_) < np.linalg.norm(LinearRegression().fit(X, y).coef_)


def test_no_intercept() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    y = X @ np.array([1.0, 2.0])
    model = LinearRegression(fit_intercept=False).fit(X, y)
    assert model.intercept_ == 0.0
    assert np.allclose(model.coef_, [1.0, 2.0])


def test_matches_sklearn(data: tuple[np.ndarray, np.ndarray]) -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")
    X, y = data
    ours = LinearRegression(l2=0.3).fit(X, y)
    ref = linear_model.Ridge(alpha=len(X) * 0.3).fit(X, y)  # alpha = n * l2
    assert np.allclose(ours.coef_, ref.coef_)
    assert ours.intercept_ == pytest.approx(ref.intercept_)
    plain = LinearRegression().fit(X, y)
    ref_plain = linear_model.LinearRegression().fit(X, y)
    assert np.allclose(plain.coef_, ref_plain.coef_)


def test_rank_deficient_does_not_crash() -> None:
    x = np.arange(10, dtype=float)[:, None]
    X = np.hstack([x, 2 * x])  # collinear columns
    model = LinearRegression().fit(X, 3 * x[:, 0])
    assert model.score(X, 3 * x[:, 0]) == pytest.approx(1.0)


def test_validation() -> None:
    with pytest.raises(ValueError):
        LinearRegression().fit(np.zeros((3, 2)), np.zeros(4))
    with pytest.raises(ValueError):
        LinearRegression().fit(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        LinearRegression().fit(np.array([[np.nan, 1.0]]), np.zeros(1))
    with pytest.raises(RuntimeError):
        LinearRegression().predict(np.zeros((1, 1)))
    with pytest.raises(ValueError):
        LinearRegression("newton")  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_divergence_is_reported() -> None:
    X = np.arange(20, dtype=float)[:, None] * 100
    with pytest.raises(FloatingPointError):
        LinearRegression("gradient_descent", learning_rate=1.0, n_iter=2000).fit(X, X[:, 0])


def _unscaled(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Features in very different units (ones, hundreds, thousands) with large offsets."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 3)) * [1.0, 100.0, 1000.0] + [0.0, 500.0, -3000.0]
    y = X @ np.array([1.0, 0.02, -0.003]) + 0.5 + rng.normal(0, 0.1, 200)
    return X, y


@pytest.mark.filterwarnings("ignore:overflow:RuntimeWarning")
def test_unscaled_features_warn_before_diverging() -> None:
    X, y = _unscaled()
    with (
        pytest.warns(DivergenceWarning, match="standardize=True"),
        pytest.raises(FloatingPointError, match="standardize=True"),
    ):
        LinearRegression("gradient_descent", n_iter=5000).fit(X, y)


def test_standardize_makes_gradient_descent_converge_on_unscaled_features() -> None:
    X, y = _unscaled()
    exact = LinearRegression().fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DivergenceWarning)
        gd = LinearRegression("gradient_descent", n_iter=5000, standardize=True).fit(X, y)
    assert np.all(np.diff(gd.loss_history_) <= 0)
    assert gd.coef_ == pytest.approx(exact.coef_, rel=1e-5)
    assert gd.intercept_ == pytest.approx(exact.intercept_, abs=1e-4)


def test_standardize_without_l2_keeps_the_least_squares_solution() -> None:
    X, y = _unscaled(1)
    plain = LinearRegression().fit(X, y)
    std = LinearRegression(standardize=True).fit(X, y)
    assert std.coef_ == pytest.approx(plain.coef_, rel=1e-9)
    assert std.intercept_ == pytest.approx(plain.intercept_, rel=1e-9)


def test_standardize_with_l2_agrees_between_methods() -> None:
    X, y = _unscaled(2)
    cf = LinearRegression(l2=0.1, standardize=True).fit(X, y)
    gd = LinearRegression("gradient_descent", l2=0.1, n_iter=5000, standardize=True).fit(X, y)
    assert gd.coef_ == pytest.approx(cf.coef_, rel=1e-6)
    assert gd.intercept_ == pytest.approx(cf.intercept_, abs=1e-4)
    # the penalty acts on standardised weights, so it differs from plain ridge
    assert not np.allclose(cf.coef_, LinearRegression(l2=0.1).fit(X, y).coef_)


def test_feature_scaling_handles_constant_columns_and_no_intercept() -> None:
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    offset, scale = feature_scaling(X, center=True)
    assert offset == pytest.approx([3.0, 5.0])
    assert scale == pytest.approx([np.sqrt(8 / 3), 1.0])
    offset, scale = feature_scaling(X, center=False)
    assert offset == pytest.approx([0.0, 0.0])
    assert scale == pytest.approx([np.sqrt(35 / 3), 5.0])

    y = 2.0 * X[:, 0]
    model = LinearRegression(
        "gradient_descent", fit_intercept=False, standardize=True, n_iter=5000, tol=0.0
    ).fit(X, y)
    assert model.intercept_ == 0.0
    exact = LinearRegression(fit_intercept=False).fit(X, y)
    assert model.coef_ == pytest.approx(exact.coef_, abs=1e-6)
