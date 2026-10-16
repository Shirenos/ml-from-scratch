"""ml-from-scratch: classic machine-learning algorithms in plain NumPy."""

from .datasets import make_blobs, make_moons, train_test_split
from .kmeans import KMeans
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .mlp import MLP
from .pca import PCA
from .softmax_regression import SoftmaxRegression

__version__ = "0.1.0"

__all__ = [
    "MLP",
    "PCA",
    "KMeans",
    "LinearRegression",
    "LogisticRegression",
    "SoftmaxRegression",
    "__version__",
    "make_blobs",
    "make_moons",
    "train_test_split",
]
