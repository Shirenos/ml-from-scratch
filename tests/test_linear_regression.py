import numpy as np
import pytest

from ml_from_scratch import LinearRegression


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
