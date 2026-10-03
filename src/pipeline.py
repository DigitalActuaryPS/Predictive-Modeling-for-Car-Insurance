"""Run every stage from raw data: `python -m src.pipeline`."""
import time

from src.data import clean
from src.features import banding
from src.interactions import glm_revision, narrative, shap_interactions
from src.models import frequency, severity
from src.impact import dislocation
from src.tariff import glm_a_mono, relativities, sensitivities


def main() -> None:
    stages = [
        ("stage 1: data and cleaning", clean.run),
        ("stage 2: banding and features", banding.run),
        ("stage 3: frequency models", frequency.run),
        ("stage 4a: SHAP interactions", shap_interactions.run),
        ("stage 4b: GLM revision", glm_revision.run),
        ("stage 4c: interaction narrative", narrative.write),
        ("stage 4d: GLM-A-mono (impact baseline)", glm_a_mono.run),
        ("stage 5: severity and burning cost", severity.run),
        ("stage 6a: tariff", relativities.run),
        ("stage 6b: exposure and BonusMalus sensitivities", sensitivities.run),
        ("stage 7: impact analysis", dislocation.run),
    ]
    for name, fn in stages:
        t = time.perf_counter()
        fn()
        print(f"{name}: {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
