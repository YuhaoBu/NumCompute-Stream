# NumCompute Stream

NumCompute Stream is an extension of the original NumCompute package.

This project implements a decision tree based, stream-compatible machine learning framework using only plain Python, NumPy, and matplotlib. It supports incremental learning from incoming data chunks, single decision tree models, bagging ensemble models, streaming metrics, streaming statistics, preprocessing updates, benchmarking, and visualisation.

## Features

### Streaming Learning

The framework supports chunk-wise updates through `.partial_fit()` or `.update()` methods.

Implemented streaming components include:

- `StandardScaler.partial_fit()`
- `Imputer.partial_fit()`
- `OneHotEncoder.partial_fit()`
- `StreamingClassificationMetrics.update()`
- `StreamingStats.update_stats()`
- `Pipeline.partial_fit()`
- `DecisionTreeClassifier.partial_fit()`
- `EnsembleClassifier.partial_fit()`
- `StreamTrainer.fit_chunk()`
- `StreamTrainer.score_chunk()`

### Decision Tree Classifier

`DecisionTreeClassifier` is implemented from scratch.

It supports:

- depth-limited tree construction
- Gini or entropy impurity
- `max_depth`
- `min_samples_split`
- `max_features`
- NaN-safe split handling
- deterministic tie resolution
- streaming adaptation through `.partial_fit()`

### Ensemble Learning

`EnsembleClassifier` implements a bagging ensemble using multiple decision trees.

It supports:

- multiple tree estimators
- bootstrap sampling
- majority voting
- `.partial_fit()` for streaming chunks
- `.predict()` for classification

### Stream Trainer

`StreamTrainer` manages model training, scoring, and logging.

It records:

- chunk accuracy
- chunk error
- cumulative accuracy
- memory footprint
- per-chunk logs

### Visualisation

The `visualise.py` module provides plotting functions using matplotlib:

- `plot_metric_over_time()`
- `compare_models()`
- `plot_predictions_vs_ground_truth()`

These functions are used in the demo notebook to show model performance over time.

## Project Structure

```text
NumCompute-Stream/
│
├── numcompute/
│   ├── io.py
│   ├── preprocessing.py
│   ├── stats.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── tree.py
│   ├── ensemble.py
│   ├── stream.py
│   ├── visualise.py
│   ├── utils.py
│   ├── benchmarking.py
│   ├── rank.py
│   ├── sort_search.py
│   └── optim.py
│
├── tests/
│   ├── test_io.py
│   ├── test_preprocessing.py
│   ├── test_stats.py
│   ├── test_metrics.py
│   ├── test_pipeline.py
│   ├── test_tree.py
│   ├── test_ensemble.py
│   ├── test_stream.py
│   ├── test_visualise.py
│   ├── test_utils.py
│   ├── test_rank.py
│   ├── test_sort_search.py
│   └── test_optim.py
│
├── benchmark/
│   └── benchmark_streaming_models.py
│
├── demo/
│   ├── stream_data.csv
│   └── stream_demo.ipynb
│
├── README.md
└── pyproject.toml
```

## Requirements

Only the following external libraries are used:

```text
numpy
matplotlib
pytest
notebook
```

No external machine learning or data processing libraries are used.

This project does not use scikit-learn, pandas, PyTorch, TensorFlow, or similar libraries.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy matplotlib pytest notebook
```

## Running Tests

Run all unit tests from the project root:

```bash
python -m pytest
```

The test suite covers:

- standard functionality
- streaming functionality
- shape mismatch handling
- NaN handling
- zero-variance scaling
- streaming metric updates
- streaming statistics
- incremental preprocessing
- decision tree prediction
- ensemble prediction
- stream trainer logging
- visualisation functions

Current test result:

```text
132 passed
```

## Running the Benchmark

The benchmark compares a single decision tree with a bagging ensemble under streaming conditions.

Run:

```bash
python -m benchmark.benchmark_streaming_models
```

Example benchmark result:

```text
Streaming Benchmark: Decision Tree vs Ensemble
==================================================
Samples: 1000
Features: 4
Chunk size: 100
Number of chunks: 10

DecisionTreeClassifier
----------------------
Final cumulative accuracy: 0.8800
Average chunk time:        0.244995 seconds
Total streaming time:      2.449953 seconds

EnsembleClassifier (Bagging)
----------------------------
Final cumulative accuracy: 0.8850
Average chunk time:        0.789073 seconds
Total streaming time:      7.890728 seconds
```

The ensemble model achieved slightly higher final accuracy, but required more computation time because it trains multiple decision trees.

## Running the Demo

Open the notebook:

```text
demo/stream_demo.ipynb
```

The demo shows:

1. Loading data from `demo/stream_data.csv` using `numcompute.io.read_csv`
2. Splitting the dataset into chunks
3. Training models incrementally using `.partial_fit()`
4. Comparing a single decision tree and a bagging ensemble
5. Logging accuracy and error over time
6. Visualising results with `visualise.py`

The main visualisations include:

- Decision Tree cumulative accuracy over time
- Decision Tree chunk error over time
- Decision Tree vs Bagging Ensemble comparison
- Predictions vs ground truth on the latest chunk

## Example Usage

### Single Decision Tree Pipeline

```python
from numcompute.preprocessing import StandardScaler
from numcompute.pipeline import Pipeline
from numcompute.tree import DecisionTreeClassifier

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=3))
])

pipe.partial_fit(X_chunk, y_chunk)
predictions = pipe.predict(X_chunk)
```

### Bagging Ensemble

```python
from numcompute.ensemble import EnsembleClassifier

model = EnsembleClassifier(
    n_estimators=5,
    method="bagging",
    max_depth=4,
    random_state=42
)

model.partial_fit(X_chunk, y_chunk)
predictions = model.predict(X_chunk)
```

### Stream Trainer

```python
from numcompute.stream import StreamTrainer

trainer = StreamTrainer(model)

trainer.fit_chunk(X_chunk, y_chunk)
log = trainer.score_chunk(X_chunk, y_chunk)

print(log)
```

Example log:

```python
{
    "chunk": 1,
    "chunk_accuracy": 0.9,
    "chunk_error": 0.1,
    "cumulative_accuracy": 0.9,
    "memory_bytes": 12345
}
```

## Design Notes

The framework is designed around consistent streaming APIs.

Models use:

```python
partial_fit(X_chunk, y_chunk)
predict(X)
```

Metrics use:

```python
update(y_true_chunk, y_pred_chunk)
result()
reset()
```

Statistics use:

```python
update_stats(X_chunk)
result()
```

This makes the package modular and allows different models, transformers, and trainers to work together in a shared pipeline.

## Numerical Stability and Edge Cases

The implementation handles:

- NaN values in preprocessing and statistics
- zero-variance columns in scaling
- shape mismatch errors
- prediction before fitting
- empty input checks
- deterministic tie resolution in voting
- chunk-wise cumulative metric updates

## AI Use Statement

Generative AI was used as an assistant to support code structuring, debugging, testing ideas, and documentation drafting. All code was reviewed, tested, and evaluated by the author. 
