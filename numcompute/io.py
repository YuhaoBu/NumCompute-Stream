import csv
import numpy as np


def _convert_cell(value, missing_values=("", "nan", "NaN", "NA", "N/A", "null", "None")):
    value = value.strip()
    if value in missing_values:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path, dtype=None, chunk_size=None, delimiter=",", skip_header=False):
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    def _reader():
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            if skip_header:
                next(reader, None)
            for row in reader:
                yield [_convert_cell(cell) for cell in row]

    if chunk_size is None:
        rows = list(_reader())
        arr = np.array(rows, dtype=object)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def _chunk_generator():
        chunk = []
        for row in _reader():
            chunk.append(row)
            if len(chunk) == chunk_size:
                arr = np.array(chunk, dtype=object)
                if dtype is not None:
                    arr = arr.astype(dtype)
                yield arr
                chunk = []
        if chunk:
            arr = np.array(chunk, dtype=object)
            if dtype is not None:
                arr = arr.astype(dtype)
            yield arr

    return _chunk_generator()