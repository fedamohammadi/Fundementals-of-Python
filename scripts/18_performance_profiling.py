"""
Performance Profiling in Python
- timing code with timeit and time.perf_counter
- profiling functions with cProfile
- memory tracking with tracemalloc
- vectorization vs loops (practical comparison)
- reading and acting on profiling results
"""

import time
import timeit


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


# ==============================================================
# Benchmarking snippets with timeit
# ==============================================================

def demo_timeit() -> None:
    loops = 1000

    t_slow = timeit.timeit("sum(range(10_000))", number=loops)
    t_fast = timeit.timeit("10_000 * (10_000 - 1) // 2", number=loops)

    print(f"sum(range(10_000)) x{loops}:  {t_slow:.4f}s")
    print(f"formula x{loops}:             {t_fast:.4f}s")
    print(f"Speedup: {t_slow / t_fast:.1f}x")


def main() -> None:
    print("=" * 50)
    print("1. time.perf_counter")
    print("=" * 50)
    demo_perf_counter()

    print()
    print("=" * 50)
    print("2. timeit")
    print("=" * 50)
    demo_timeit()


if __name__ == "__main__":
    main()
