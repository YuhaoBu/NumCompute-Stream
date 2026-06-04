import time
import numpy as np

from numcompute.tree import DecisionTreeClassifier
from numcompute.ensemble import EnsembleClassifier
from numcompute.stream import StreamTrainer


def make_synthetic_dataset(n_samples=1000, n_features=4, random_state=42):
    rng = np.random.default_rng(random_state)

    X = rng.normal(size=(n_samples, n_features))
    scores = X[:, 0] + 0.5 * X[:, 1] - 0.25 * X[:, 2]
    y = (scores > 0).astype(int)

    noise_idx = rng.choice(n_samples, size=n_samples // 10, replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    return X, y


def make_chunks(X, y, chunk_size=100):
    for start in range(0, X.shape[0], chunk_size):
        end = start + chunk_size
        yield X[start:end], y[start:end]


def run_streaming_benchmark(model, X, y, chunk_size=100):
    trainer = StreamTrainer(model)

    chunk_accuracies = []
    cumulative_accuracies = []
    chunk_times = []

    for X_chunk, y_chunk in make_chunks(X, y, chunk_size):
        start = time.time()

        log = trainer.fit_score_chunk(X_chunk, y_chunk)

        end = time.time()

        chunk_accuracies.append(log["chunk_accuracy"])
        cumulative_accuracies.append(log["cumulative_accuracy"])
        chunk_times.append(end - start)

    return {
        "chunk_accuracies": chunk_accuracies,
        "cumulative_accuracies": cumulative_accuracies,
        "chunk_times": chunk_times,
        "avg_time": float(np.mean(chunk_times)),
        "total_time": float(np.sum(chunk_times)),
        "final_accuracy": float(cumulative_accuracies[-1]),
    }


def print_result(name, result):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Final cumulative accuracy: {result['final_accuracy']:.4f}")
    print(f"Average chunk time:        {result['avg_time']:.6f} seconds")
    print(f"Total streaming time:      {result['total_time']:.6f} seconds")


def main():
    X, y = make_synthetic_dataset(
        n_samples=1000,
        n_features=4,
        random_state=42,
    )

    chunk_size = 100

    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=2,
        max_features=None,
        criterion="gini",
    )

    ensemble = EnsembleClassifier(
        n_estimators=5,
        method="bagging",
        max_depth=4,
        min_samples_split=2,
        max_features=None,
        criterion="gini",
        random_state=42,
    )

    tree_result = run_streaming_benchmark(tree, X, y, chunk_size)
    ensemble_result = run_streaming_benchmark(ensemble, X, y, chunk_size)

    print("\nStreaming Benchmark: Decision Tree vs Ensemble")
    print("=" * 50)
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of chunks: {len(tree_result['chunk_accuracies'])}")

    print_result("DecisionTreeClassifier", tree_result)
    print_result("EnsembleClassifier (Bagging)", ensemble_result)


if __name__ == "__main__":
    main()