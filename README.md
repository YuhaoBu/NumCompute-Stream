# NumCompute

A modular scientific computing toolkit implemented using Python and NumPy.
It includes components for data loading, preprocessing, statistics, sorting/searching, ranking, evaluation metrics, numerical optimisation, and pipeline composition.

---

## Installation

```bash
git clone https://github.com/Luxf23/NumCompute
cd NumCompute
pip install numpy pytest
```

---

## Project Structure

```
numcompute/
  io.py
  preprocessing.py
  sort_search.py
  rank.py
  stats.py
  metrics.py
  optim.py
  pipeline.py
  utils.py

tests/
demo/
```

---

## API Overview

**IO**

* read_csv(path, skip_header=False, chunk_size=None)

**Preprocessing**

* StandardScaler
* MinMaxScaler
* OneHotEncoder
* Imputer

**Statistics**

* mean
* var
* quantile
* histogram
* Welford

**Sort & Search**

* stable_sort
* multi_key_sort
* topk
* quickselect
* binary_search

**Rank**

* rank(data, method)
* percentile(data, q)

**Metrics**

* accuracy
* precision
* recall
* f1
* confusion_matrix
* mse

**Optimisation**

* grad(f, x)
* jacobian(F, x)

**Pipeline**

* Pipeline([...])

---

## Performance

Core computations are vectorised using NumPy.
Vectorised operations are significantly faster than Python loops.

---

## Testing

```bash
python -m pytest
```

---

## Demo

A complete example is provided in:

```
demo/quickstart.ipynb
```

It demonstrates:

* CSV loading and preprocessing
* Sorting, ranking, and statistics
* Evaluation metrics
* Gradient and Jacobian computation
* Pipeline chaining
* Performance comparison

---

## Notes

* Implemented using only Python and NumPy
* No external machine learning or database libraries are used
* Designed for clarity and modularity
