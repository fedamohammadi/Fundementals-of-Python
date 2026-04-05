"""
Performance Profiling in Python
- timing code with timeit and time.perf_counter
- profiling functions with cProfile
- memory tracking with tracemalloc
- vectorization vs loops (practical comparison)
- reading and acting on profiling results
"""

import time


# ==============================================================
# Timing with time.perf_counter
# ==============================================================

def slow_sum(n: int) -> int:
    total = 0
    for i in range(n):
        total += i
    return total


def fast_sum(n: int) -> int:
    return n * (n - 1) // 2


def demo_perf_counter() -> None:
    n = 1_000_000

    start = time.perf_counter()
    slow_sum(n)
    elapsed = time.perf_counter() - start
    print(f"slow_sum({n:,}):  {elapsed:.6f}s")

    start = time.perf_counter()
    fast_sum(n)
    elapsed = time.perf_counter() - start
    print(f"fast_sum({n:,}):  {elapsed:.6f}s")


def main() -> None:
    print("=" * 50)
    print("1. time.perf_counter")
    print("=" * 50)
    demo_perf_counter()


if __name__ == "__main__":
    main()
