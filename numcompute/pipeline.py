class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X):
        for name, step in self.steps:
            if not hasattr(step, "fit"):
                raise AttributeError(f"{name} has no fit method")

            step.fit(X)

            if hasattr(step, "transform"):
                X = step.transform(X)

        return self

    def transform(self, X):
        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise AttributeError(f"{name} has no transform method")

            X = step.transform(X)

        return X

    def fit_transform(self, X):
        for name, step in self.steps:
            if hasattr(step, "fit_transform"):
                X = step.fit_transform(X)
            else:
                if not hasattr(step, "fit") or not hasattr(step, "transform"):
                    raise AttributeError(f"{name} must have fit and transform")

                step.fit(X)
                X = step.transform(X)

        return X
