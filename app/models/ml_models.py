"""Small numpy-only regressors used by the forecaster.

They live here (not in the training script) because joblib pickles classes by
module path; defined in __main__ the saved model couldn't be loaded elsewhere.
Don't rename or move these classes or their attributes without retraining.
"""

import numpy as np


class RidgeRegression:
    """L2-regularised linear regression with built-in feature standardisation."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8  # constant columns would divide by zero
        Xn = np.c_[(X - self.mu) / self.sigma, np.ones(len(X))]

        A = Xn.T @ Xn + self.alpha * np.eye(Xn.shape[1])
        self.weights = np.linalg.solve(A, Xn.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = np.c_[(X - self.mu) / self.sigma, np.ones(len(X))]
        return Xn @ self.weights


class DecisionStump:
    """Depth-1 tree: one threshold on one feature."""

    def __init__(self):
        self.feature_idx: int = 0
        self.threshold: float = 0.0
        self.left_value: float = 0.0
        self.right_value: float = 0.0

    def fit(self, X: np.ndarray, residuals: np.ndarray) -> "DecisionStump":
        """Pick the split with the lowest squared error over a random feature subset."""
        best_loss = float("inf")
        n_feat = X.shape[1]
        feat_indices = np.random.choice(n_feat, max(1, int(np.sqrt(n_feat))), replace=False)

        for f in feat_indices:
            col = X[:, f]
            for t in np.percentile(col, np.linspace(10, 90, 10)):
                left_mask = col <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                lv = residuals[left_mask].mean()
                rv = residuals[right_mask].mean()
                loss = (((residuals[left_mask] - lv) ** 2).sum() +
                        ((residuals[right_mask] - rv) ** 2).sum())

                if loss < best_loss:
                    best_loss = loss
                    self.feature_idx = f
                    self.threshold = t
                    self.left_value = lv
                    self.right_value = rv

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.feature_idx]
        return np.where(col <= self.threshold, self.left_value, self.right_value)


class GradientBoostingRegressor:
    """Gradient boosting over decision stumps, with row subsampling per round."""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.stumps: list[DecisionStump] = []
        self.initial_prediction: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        np.random.seed(self.random_state)

        self.initial_prediction = y.mean()
        F = np.full(len(y), self.initial_prediction)
        n = len(y)
        subsample_size = int(n * self.subsample)

        for _ in range(self.n_estimators):
            residuals = y - F
            idx = np.random.choice(n, subsample_size, replace=False)
            stump = DecisionStump().fit(X[idx], residuals[idx])
            self.stumps.append(stump)
            F += self.learning_rate * stump.predict(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = np.full(len(X), self.initial_prediction)
        for stump in self.stumps:
            F += self.learning_rate * stump.predict(X)
        return F
