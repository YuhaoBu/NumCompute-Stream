import numpy as np
from numcompute.io import read_csv


def test_read_csv_basic(tmp_path):
    file_path = tmp_path / "test.csv"
    file_path.write_text("1,2\n3,4\n", encoding="utf-8")

    arr = read_csv(file_path)

    assert arr.shape == (2, 2)
    assert arr[0, 0] == 1.0
    assert arr[1, 1] == 4.0


def test_read_csv_with_missing_values(tmp_path):
    file_path = tmp_path / "test_missing.csv"
    file_path.write_text("1,\n3,4\n", encoding="utf-8")

    arr = read_csv(file_path)

    assert arr.shape == (2, 2)
    assert arr[0, 0] == 1.0
    assert np.isnan(arr[0, 1])
    assert arr[1, 1] == 4.0


def test_read_csv_skip_header(tmp_path):
    file_path = tmp_path / "test_header.csv"
    file_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    arr = read_csv(file_path, skip_header=True)

    assert arr.shape == (2, 2)
    assert arr[0, 0] == 1.0
    assert arr[1, 1] == 4.0


def test_read_csv_chunked(tmp_path):
    file_path = tmp_path / "test_chunk.csv"
    file_path.write_text("1,2\n3,4\n5,6\n", encoding="utf-8")

    chunks = list(read_csv(file_path, chunk_size=2))

    assert len(chunks) == 2
    assert chunks[0].shape == (2, 2)
    assert chunks[1].shape == (1, 2)
    assert chunks[1][0, 0] == 5.0