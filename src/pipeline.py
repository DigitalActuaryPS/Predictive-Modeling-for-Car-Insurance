"""Run every stage from raw data: `python -m src.pipeline`."""
import time

from src.data import clean
from src.features import banding


def main() -> None:
    stages = [
        ("stage 1: data and cleaning", clean.run),
        ("stage 2: banding and features", banding.run),
    ]
    for name, fn in stages:
        t = time.perf_counter()
        fn()
        print(f"{name}: {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
