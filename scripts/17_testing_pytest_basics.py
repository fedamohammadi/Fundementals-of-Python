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




if __name__ == "__main__":
    main()
