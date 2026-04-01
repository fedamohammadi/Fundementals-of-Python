"""
- simple test functions
- assertions for testing
- testing edge cases
"""


def add(a: int, b: int) -> int:
    return a + b


def main() -> None:
    print("Run tests by calling test functions manually.")


# ==============================================================
# Simple test functions
# ==============================================================

def test_add() -> None:
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def main() -> None:
    print("Running basic tests...")
    test_add()
    print("All tests passed.")


# ==============================================================
# More realistic example + edge cases
# ==============================================================

def compute_average(values: list[float]) -> float:
    if len(values) == 0:
        raise ValueError("Empty list")
    return sum(values) / len(values)


def test_compute_average() -> None:
    assert compute_average([10.0, 20.0, 30.0]) == 20.0

    # Edge case: single value
    assert compute_average([5.0]) == 5.0

    # Edge case: empty list should raise error
    try:
        compute_average([])
        assert False  # Should not reach here
    except ValueError:
        assert True


def main() -> None:
    test_add()
    test_compute_average()
    print("All tests passed.")





if __name__ == "__main__":
    main()
