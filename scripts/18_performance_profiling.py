"""
Performance Profiling in Python
- timing code with timeit and time.perf_counter
- profiling functions with cProfile
- memory tracking with tracemalloc
- vectorization vs loops (practical comparison)
- reading and acting on profiling results
"""

import cProfile
import io
import pstats
import time
import timeit
import tracemalloc


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


# ==============================================================
# Profiling functions with cProfile
# ==============================================================

def load_data(n: int) -> list[float]:
    return [i * 0.5 for i in range(n)]


def compute_stats(data: list[float]) -> dict:
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return {"mean": mean, "variance": variance}


def analysis_pipeline(n: int = 100_000) -> dict:
    data = load_data(n)
    return compute_stats(data)


def demo_cprofile() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    analysis_pipeline()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(8)
    print(stream.getvalue())


# ==============================================================
# Memory tracking with tracemalloc
# ==============================================================

def build_list(n: int) -> list[int]:
    return list(range(n))


def build_set(n: int) -> set[int]:
    return set(range(n))


def demo_tracemalloc() -> None:
    n = 500_000

    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    data = build_list(n)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top = after.compare_to(before, "lineno")[0]
    print(f"build_list({n:,}): +{top.size_diff / 1024:.1f} KB  ({top.count_diff} new objects)")

    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    _ = build_set(n)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top = after.compare_to(before, "lineno")[0]
    print(f"build_set({n:,}):  +{top.size_diff / 1024:.1f} KB  ({top.count_diff} new objects)")

    del data


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

    print()
    print("=" * 50)
    print("3. cProfile")
    print("=" * 50)
    demo_cprofile()

    print()
    print("=" * 50)
    print("4. tracemalloc")
    print("=" * 50)
    demo_tracemalloc()


if __name__ == "__main__":
    main()
