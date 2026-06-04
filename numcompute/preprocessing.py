import numpy as np


def _to_2d_float_array(X, name="X"):
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    return arr


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self.n_seen_ = None
        self.M2_ = None

    def fit(self, X):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self.n_seen_ = None
        self.M2_ = None
        return self.partial_fit(X)

    def partial_fit(self, X):
        X = _to_2d_float_array(X)

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.n_seen_ = np.zeros(X.shape[1], dtype=int)
            self.mean_ = np.zeros(X.shape[1], dtype=float)
            self.M2_ = np.zeros(X.shape[1], dtype=float)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but StandardScaler was fitted with "
                f"{self.n_features_in_} features."
            )

        for row in X:
            mask = ~np.isnan(row)
            values = row[mask]

            old_count = self.n_seen_[mask]
            new_count = old_count + 1

            delta = values - self.mean_[mask]
            self.mean_[mask] += delta / new_count
            delta2 = values - self.mean_[mask]
            self.M2_[mask] += delta * delta2
            self.n_seen_[mask] = new_count

        variance = np.zeros_like(self.mean_)
        valid = self.n_seen_ > 0
        variance[valid] = self.M2_[valid] / self.n_seen_[valid]

        scale = np.sqrt(variance)
        self.scale_ = np.where(scale == 0, 1.0, scale)

        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler must be fitted before calling transform().")

        X = _to_2d_float_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but StandardScaler was fitted with "
                f"{self.n_features_in_} features."
            )
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:

    def __init__(self):
        self.min_ = None
        self.range_ = None
        self.n_features_in_ = None

    def fit(self, X):
        X = _to_2d_float_array(X)
        self.min_ = np.nanmin(X, axis=0)
        max_ = np.nanmax(X, axis=0)
        range_ = max_ - self.min_
        self.range_ = np.where(range_ == 0, 1.0, range_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if self.min_ is None or self.range_ is None:
            raise ValueError("MinMaxScaler must be fitted before calling transform().")

        X = _to_2d_float_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but MinMaxScaler was fitted with "
                f"{self.n_features_in_} features."
            )
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Imputer:

    def __init__(self, strategy="mean"):
        valid = {"mean", "median", "most_frequent"}
        if strategy not in valid:
            raise ValueError(f"strategy must be one of {valid}, got {strategy!r}.")
        self.strategy = strategy
        self.fill_values_ = None
        self.n_features_in_ = None
        self._sum_ = None
        self._count_ = None
        self._values_ = None
        self._freqs_ = None

    def fit(self, X):
        self.fill_values_ = None
        self.n_features_in_ = None
        self._sum_ = None
        self._count_ = None
        self._values_ = None
        self._freqs_ = None
        return self.partial_fit(X)

    def partial_fit(self, X):
        X = _to_2d_float_array(X)

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]

            if self.strategy == "mean":
                self._sum_ = np.zeros(X.shape[1], dtype=float)
                self._count_ = np.zeros(X.shape[1], dtype=int)

            elif self.strategy == "median":
                self._values_ = [np.array([], dtype=float) for _ in range(X.shape[1])]

            else:
                self._freqs_ = [dict() for _ in range(X.shape[1])]

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but Imputer was fitted with "
                f"{self.n_features_in_} features."
            )

        if self.strategy == "mean":
            valid = ~np.isnan(X)
            self._sum_ += np.nansum(X, axis=0)
            self._count_ += np.sum(valid, axis=0)

            fill_values = np.divide(
                self._sum_,
                self._count_,
                out=np.zeros_like(self._sum_),
                where=self._count_ > 0
            )

        elif self.strategy == "median":
            fill_values = []

            for j in range(X.shape[1]):
                valid = X[:, j][~np.isnan(X[:, j])]

                if valid.size > 0:
                    self._values_[j] = np.concatenate([self._values_[j], valid])

                if self._values_[j].size == 0:
                    fill_values.append(0.0)
                else:
                    fill_values.append(np.median(self._values_[j]))

            fill_values = np.asarray(fill_values, dtype=float)

        else:
            fill_values = []

            for j in range(X.shape[1]):
                valid = X[:, j][~np.isnan(X[:, j])]

                for value in valid:
                    self._freqs_[j][value] = self._freqs_[j].get(value, 0) + 1

                if len(self._freqs_[j]) == 0:
                    fill_values.append(0.0)
                else:
                    best_value = min(
                        self._freqs_[j].items(),
                        key=lambda item: (-item[1], item[0])
                    )[0]
                    fill_values.append(best_value)

            fill_values = np.asarray(fill_values, dtype=float)

        self.fill_values_ = np.where(np.isnan(fill_values), 0.0, fill_values)
        return self

    def transform(self, X):
        if self.fill_values_ is None:
            raise ValueError("Imputer must be fitted before calling transform().")

        X = _to_2d_float_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but Imputer was fitted with "
                f"{self.n_features_in_} features."
            )

        out = X.copy()
        nan_mask = np.isnan(out)
        if np.any(nan_mask):
            out[nan_mask] = np.take(self.fill_values_, np.where(nan_mask)[1])
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:

    def __init__(self, handle_unknown="error"):
        valid = {"error", "ignore"}
        if handle_unknown not in valid:
            raise ValueError(f"handle_unknown must be one of {valid}, got {handle_unknown!r}.")
        self.handle_unknown = handle_unknown
        self.categories_ = None
        self.category_to_index_ = None

    def fit(self, x):
        self.categories_ = None
        self.category_to_index_ = None
        return self.partial_fit(x)

    def partial_fit(self, x):
        arr = np.asarray(x)
        if arr.ndim != 1:
            raise ValueError(f"x must be a 1D array, got shape {arr.shape}.")

        new_categories = np.unique(arr)

        if self.categories_ is None:
            self.categories_ = new_categories
        else:
            self.categories_ = np.unique(
                np.concatenate([self.categories_, new_categories])
            )

        self.category_to_index_ = {
            cat: idx for idx, cat in enumerate(self.categories_)
        }

        return self

    def transform(self, x):
        if self.categories_ is None:
            raise ValueError("OneHotEncoder must be fitted before calling transform().")

        arr = np.asarray(x)
        if arr.ndim != 1:
            raise ValueError(f"x must be a 1D array, got shape {arr.shape}.")

        n_samples = arr.shape[0]
        n_categories = len(self.categories_)
        out = np.zeros((n_samples, n_categories), dtype=int)

        for i, value in enumerate(arr):
            idx = self.category_to_index_.get(value)
            if idx is None:
                if self.handle_unknown == "error":
                    raise ValueError(f"Unknown category {value!r} encountered during transform().")
                continue
            out[i, idx] = 1

        return out

    def fit_transform(self, x):
        return self.fit(x).transform(x)