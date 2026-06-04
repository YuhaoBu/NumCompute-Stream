import numpy as np

from numcompute.tree import DecisionTreeClassifier


class EnsembleClassifier:
    def __init__(
        self,
        n_estimators=5,
        method="bagging",
        max_depth=3,
        min_samples_split=2,
        max_features=None,
        criterion="gini",
        random_state=None,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")

        if method != "bagging":
            raise ValueError("Only 'bagging' is currently supported.")

        self.n_estimators = n_estimators
        self.method = method
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state

        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                criterion=criterion,
            )
            for _ in range(n_estimators)
        ]

        self.rng_ = np.random.default_rng(random_state)
        self.classes_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk, y_chunk = self._validate_X_y(X_chunk, y_chunk)

        if self.n_features_in_ is None:
            self.n_features_in_ = X_chunk.shape[1]
        elif X_chunk.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_chunk has {X_chunk.shape[1]} features, but EnsembleClassifier "
                f"was fitted with {self.n_features_in_} features."
            )

        if self.classes_ is None:
            self.classes_ = np.unique(y_chunk)
        else:
            self.classes_ = np.unique(np.concatenate([self.classes_, y_chunk]))

        n_samples = X_chunk.shape[0]

        for estimator in self.estimators_:
            bootstrap_indices = self.rng_.integers(
                low=0,
                high=n_samples,
                size=n_samples,
            )

            X_bootstrap = X_chunk[bootstrap_indices]
            y_bootstrap = y_chunk[bootstrap_indices]

            estimator.partial_fit(X_bootstrap, y_bootstrap)

        self.is_fitted_ = True
        return self

    def fit(self, X, y):
        self._reset_estimators()
        return self.partial_fit(X, y)

    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("EnsembleClassifier must be fitted before predict().")

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but EnsembleClassifier was fitted "
                f"with {self.n_features_in_} features."
            )

        all_predictions = np.array([
            estimator.predict(X)
            for estimator in self.estimators_
        ])

        return np.array([
            self._majority_vote(all_predictions[:, i])
            for i in range(X.shape[0])
        ])

    def _reset_estimators(self):
        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                criterion=self.criterion,
            )
            for _ in range(self.n_estimators)
        ]

        self.classes_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False

    def _majority_vote(self, votes):
        values, counts = np.unique(votes, return_counts=True)
        return values[np.argmax(counts)]

    def _validate_X(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")

        return X

    def _validate_X_y(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array, got shape {y.shape}.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        if X.shape[0] == 0:
            raise ValueError("X and y must not be empty.")

        return X, y