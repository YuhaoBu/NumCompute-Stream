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

    def partial_fit(self, X, y=None):
        if len(self.steps) == 0:
            raise ValueError("Pipeline must contain at least one step.")

        for name, step in self.steps[:-1]:
            if not hasattr(step, "partial_fit"):
                raise AttributeError(f"{name} has no partial_fit method")

            step.partial_fit(X)

            if not hasattr(step, "transform"):
                raise AttributeError(f"{name} has no transform method")

            X = step.transform(X)

        final_name, final_step = self.steps[-1]

        if not hasattr(final_step, "partial_fit"):
            raise AttributeError(f"{final_name} has no partial_fit method")

        if y is None:
            final_step.partial_fit(X)
        else:
            final_step.partial_fit(X, y)

        return self

    def predict(self, X):
        if len(self.steps) == 0:
            raise ValueError("Pipeline must contain at least one step.")

        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise AttributeError(f"{name} has no transform method")

            X = step.transform(X)

        final_name, final_step = self.steps[-1]

        if not hasattr(final_step, "predict"):
            raise AttributeError(f"{final_name} has no predict method")

        return final_step.predict(X)
