"""Run every stage from raw data: `python -m src.pipeline`."""
import os
import platform
import time

import pandas as pd

from src.config import load_config
from src.data import clean
from src.features import banding
from src.impact import dislocation
from src.interactions import glm_revision, narrative, shap_interactions
from src.models import frequency, severity
from src.reporting import documents
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
        ("stage 8: README and LIMITATIONS", documents.run),
    ]
    rows = []
    for name, fn in stages:
        t = time.perf_counter()
        fn()
        secs = time.perf_counter() - t
        rows.append({"stage": name, "seconds": round(secs, 1)})
        print(f"{name}: {secs:.1f}s", flush=True)
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {"stage": f"total ({os.cpu_count()} CPUs, Python {platform.python_version()})",
                         "seconds": round(out["seconds"].sum(), 1)}
    out.to_csv(load_config()["paths"]["tables"] / "pipeline_runtime.csv", index=False)


if __name__ == "__main__":
    main()
