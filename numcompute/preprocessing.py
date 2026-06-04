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

    def fit(self, X):
        X = _to_2d_float_array(X)
        self.mean_ = np.nanmean(X, axis=0)
        scale = np.nanstd(X, axis=0)
        self.scale_ = np.where(scale == 0, 1.0, scale)
        self.n_features_in_ = X.shape[1]
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

    def fit(self, X):
        X = _to_2d_float_array(X)
        self.n_features_in_ = X.shape[1]

        if self.strategy == "mean":
            fill_values = np.nanmean(X, axis=0)

        elif self.strategy == "median":
            fill_values = np.nanmedian(X, axis=0)

        else:
            fill_values = []
            for col in X.T:
                valid = col[~np.isnan(col)]
                if valid.size == 0:
                    fill_values.append(0.0)
                else:
                    values, counts = np.unique(valid, return_counts=True)
                    fill_values.append(values[np.argmax(counts)])
            fill_values = np.asarray(fill_values, dtype=float)

        fill_values = np.where(np.isnan(fill_values), 0.0, fill_values)
        self.fill_values_ = fill_values
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
        arr = np.asarray(x)
        if arr.ndim != 1:
            raise ValueError(f"x must be a 1D array, got shape {arr.shape}.")

        categories = np.unique(arr)
        self.categories_ = categories
        self.category_to_index_ = {cat: idx for idx, cat in enumerate(categories)}
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