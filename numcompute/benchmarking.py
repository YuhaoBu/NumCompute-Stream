"""
benchmarking.py

Simple benchmarking utilities.
"""

import time


def benchmark(func, *args, repeats=5):
    """
    Measure execution time.
    """
    times = []

    for _ in range(repeats):
        start = time.time()
        func(*args)
        end = time.time()
        times.append(end - start)

    return {
        "min": min(times),
        "max": max(times),
        "avg": sum(times) / len(times)
    }