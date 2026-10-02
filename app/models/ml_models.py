"""
app/models/ml_models.py
========================
Pure-numpy ML model implementations shared by both:
  - training/train_forecaster.py  (trains and saves)
  - app/models/forecasting.py     (loads and runs inference)

WHY a shared module?
--------------------
When you save a Python object with joblib/pickle, it stores the CLASS
by its fully-qualified module path (e.g. "app.models.ml_models.RidgeRegression").
When you load it back, Python must be able to import that exact path to
reconstruct the object.

If the class was defined in __main__ (a script), it saves as "__main__.RidgeRegression"
which is NOT resolvable from any other module. Keeping model classes here
ensures they always resolve correctly regardless of where load() is called.

Classes:
    RidgeRegression            — L2-regularised linear regression (pure numpy)
    GradientBoostingRegressor  — gradient boosted depth-1 stumps (pure numpy)
    DecisionStump              — single binary split weak learner
"""

import numpy as np


# =============================================================================
# Ridge Regression
# =============================================================================

class RidgeRegression:
    """
    Ridge Regression — L2-regularised linear regression, pure numpy.

    THEORY
    ------
    Minimises:  ||y - Xw||² + α||w||²

    Closed-form solution (Normal Equations with ridge):
        w = (XᵀX + αI)⁻¹ Xᵀy

    Features are standardised internally (zero mean, unit variance).
    A bias (intercept) column of 1s is appended before solving.

    Parameters
    ----------
    alpha : float
        Regularisation strength. Higher = more shrinkage.
        α=0 reduces to ordinary least squares.
        α=1 is a robust default.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha  = alpha
        self.weights: np.ndarray | None = None
        self.mu: np.ndarray | None     = None   # training feature means
        self.sigma: np.ndarray | None  = None   # training feature stds

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """Fit the model. Stores weights, mu, sigma for later predict()."""
        self.mu    = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8       # avoid div-by-zero
        Xn = (X - self.mu) / self.sigma

        # Add bias column
        Xn = np.c_[Xn, np.ones(len(Xn))]

        n_features = Xn.shape[1]
        A = Xn.T @ Xn + self.alpha * np.eye(n_features)
        b = Xn.T @ y

        # np.linalg.solve is more stable than explicit matrix inversion
        self.weights = np.linalg.solve(A, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions. Applies same standardisation as training."""
        Xn = (X - self.mu) / self.sigma
        Xn = np.c_[Xn, np.ones(len(Xn))]
        return Xn @ self.weights


# =============================================================================
# Decision Stump (weak learner for gradient boosting)
# =============================================================================

class DecisionStump:
    """
    A depth-1 decision tree — the simplest non-trivial model.
    Makes a single binary split on one feature at one threshold.
    Used as the weak learner in GradientBoostingRegressor.
    """

    def __init__(self):
        self.feature_idx: int   = 0
        self.threshold:   float = 0.0
        self.left_value:  float = 0.0
        self.right_value: float = 0.0

    def fit(self, X: np.ndarray, residuals: np.ndarray) -> "DecisionStump":
        """Find the split that minimises squared error on residuals."""
        best_loss = float("inf")
        n, n_feat = X.shape

        n_feat_sample = max(1, int(np.sqrt(n_feat)))
        feat_indices  = np.random.choice(n_feat, n_feat_sample, replace=False)

        for f in feat_indices:
            col        = X[:, f]
            thresholds = np.percentile(col, np.linspace(10, 90, 10))

            for t in thresholds:
                left_mask  = col <= t
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                lv   = residuals[left_mask].mean()
                rv   = residuals[right_mask].mean()
                loss = (((residuals[left_mask]  - lv) ** 2).sum() +
                        ((residuals[right_mask] - rv) ** 2).sum())

                if loss < best_loss:
                    best_loss        = loss
                    self.feature_idx = f
                    self.threshold   = t
                    self.left_value  = lv
                    self.right_value = rv

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.feature_idx]
        return np.where(col <= self.threshold, self.left_value, self.right_value)


# =============================================================================
# Gradient Boosting Regressor
# =============================================================================

class GradientBoostingRegressor:
    """
    Gradient Boosted Decision Stumps — pure numpy.

    Builds an additive ensemble by iteratively fitting stumps to residuals.
    Each stump corrects the errors of the previous ensemble.

    Parameters
    ----------
    n_estimators  : number of boosting rounds
    learning_rate : shrinkage factor per round (smaller = more conservative)
    subsample     : fraction of training data used per round (reduces overfit)
    random_state  : seed for reproducibility
    """

    def __init__(
        self,
        n_estimators:  int   = 200,
        learning_rate: float = 0.1,
        subsample:     float = 0.8,
        random_state:  int   = 42,
    ):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self.random_state  = random_state
        self.stumps: list[DecisionStump] = []
        self.initial_prediction: float   = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        """Train the boosted ensemble."""
        np.random.seed(self.random_state)

        self.initial_prediction = y.mean()
        F = np.full(len(y), self.initial_prediction)
        n = len(y)
        subsample_size = int(n * self.subsample)

        for i in range(self.n_estimators):
            residuals = y - F
            idx       = np.random.choice(n, subsample_size, replace=False)
            stump     = DecisionStump().fit(X[idx], residuals[idx])
            self.stumps.append(stump)
            F += self.learning_rate * stump.predict(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions as the sum of all stumps."""
        F = np.full(len(X), self.initial_prediction)
        for stump in self.stumps:
            F += self.learning_rate * stump.predict(X)
        return F
