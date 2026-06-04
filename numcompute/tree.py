import numpy as np


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        max_features=None,
        criterion="gini",
    ):
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be either 'gini' or 'entropy'.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion

        self.root_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self._X_seen = None
        self._y_seen = None

    def fit(self, X, y):
        X, y = self._validate_X_y(X, y)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._X_seen = X.copy()
        self._y_seen = y.copy()
        self.root_ = self._build_tree(X, y, depth=0)

        return self

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk, y_chunk = self._validate_X_y(X_chunk, y_chunk)

        if self.n_features_in_ is None:
            self.n_features_in_ = X_chunk.shape[1]
            self._X_seen = X_chunk.copy()
            self._y_seen = y_chunk.copy()
            self.classes_ = np.unique(y_chunk)
        else:
            if X_chunk.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X_chunk has {X_chunk.shape[1]} features, but the tree was "
                    f"initialised with {self.n_features_in_} features."
                )

            self._X_seen = np.vstack([self._X_seen, X_chunk])
            self._y_seen = np.concatenate([self._y_seen, y_chunk])
            self.classes_ = np.unique(np.concatenate([self.classes_, y_chunk]))

        self.root_ = self._build_tree(self._X_seen, self._y_seen, depth=0)
        return self

    def predict(self, X):
        if self.root_ is None:
            raise ValueError("DecisionTreeClassifier must be fitted before predict().")

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the tree was fitted with "
                f"{self.n_features_in_} features."
            )

        return np.array([self._predict_one(row, self.root_) for row in X])

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

    def _build_tree(self, X, y, depth):
        prediction = self._majority_class(y)

        node = {
            "is_leaf": True,
            "prediction": prediction,
            "feature": None,
            "threshold": None,
            "left": None,
            "right": None,
            "nan_go_to": "left",
        }

        if depth >= self.max_depth:
            return node

        if X.shape[0] < self.min_samples_split:
            return node

        if np.unique(y).size == 1:
            return node

        split = self._best_split(X, y)

        if split is None:
            return node

        feature, threshold, nan_go_to = split
        values = X[:, feature]
        nan_mask = np.isnan(values)

        left_mask = values <= threshold
        right_mask = values > threshold

        if nan_go_to == "left":
            left_mask = left_mask | nan_mask
        else:
            right_mask = right_mask | nan_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return node

        node["is_leaf"] = False
        node["feature"] = feature
        node["threshold"] = threshold
        node["nan_go_to"] = nan_go_to
        node["left"] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node["right"] = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _best_split(self, X, y):
        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_split = None

        feature_indices = self._feature_indices(X.shape[1])

        for feature in feature_indices:
            values = X[:, feature]
            valid_values = values[~np.isnan(values)]

            if valid_values.size == 0:
                continue

            unique_values = np.unique(valid_values)

            if unique_values.size <= 1:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                nan_mask = np.isnan(values)
                left_mask = values <= threshold
                right_mask = values > threshold

                if np.sum(left_mask) >= np.sum(right_mask):
                    nan_go_to = "left"
                    left_mask = left_mask | nan_mask
                else:
                    nan_go_to = "right"
                    right_mask = right_mask | nan_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left == 0 or n_right == 0:
                    continue

                gain = parent_impurity
                gain -= (n_left / y.size) * self._impurity(y[left_mask])
                gain -= (n_right / y.size) * self._impurity(y[right_mask])

                if best_split is None or gain > best_gain + 1e-12:
                    best_gain = gain
                    best_split = (feature, threshold, nan_go_to)

                elif np.isclose(gain, best_gain):
                    if best_split is None:
                        best_split = (feature, threshold, nan_go_to)
                    else:
                        old_feature, old_threshold, _ = best_split
                        if feature < old_feature or (
                            feature == old_feature and threshold < old_threshold
                        ):
                            best_split = (feature, threshold, nan_go_to)

        return best_split

    def _feature_indices(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)

        if isinstance(self.max_features, int):
            k = min(max(self.max_features, 1), n_features)
            return np.arange(k)

        if isinstance(self.max_features, float):
            if not 0 < self.max_features <= 1:
                raise ValueError("float max_features must be in the interval (0, 1].")
            k = max(1, int(np.ceil(self.max_features * n_features)))
            return np.arange(k)

        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
            return np.arange(k)

        if self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
            return np.arange(k)

        raise ValueError("max_features must be None, int, float, 'sqrt', or 'log2'.")

    def _impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()

        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)

        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _predict_one(self, row, node):
        while not node["is_leaf"]:
            feature = node["feature"]
            threshold = node["threshold"]

            if np.isnan(row[feature]):
                if node["nan_go_to"] == "left":
                    node = node["left"]
                else:
                    node = node["right"]
            elif row[feature] <= threshold:
                node = node["left"]
            else:
                node = node["right"]

        return node["prediction"]