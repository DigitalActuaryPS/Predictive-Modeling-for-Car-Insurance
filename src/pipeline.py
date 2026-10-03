"""Run every stage from raw data: `python -m src.pipeline`."""
import time

from src.data import clean
from src.features import banding
from src.interactions import glm_revision, narrative, shap_interactions
from src.models import frequency


def main() -> None:
    stages = [
        ("stage 1: data and cleaning", clean.run),
        ("stage 2: banding and features", banding.run),
        ("stage 3: frequency models", frequency.run),
        ("stage 4a: SHAP interactions", shap_interactions.run),
        ("stage 4b: GLM revision", glm_revision.run),
        ("stage 4c: interaction narrative", narrative.write),
    ]
    for name, fn in stages:
        t = time.perf_counter()
        fn()
        print(f"{name}: {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
