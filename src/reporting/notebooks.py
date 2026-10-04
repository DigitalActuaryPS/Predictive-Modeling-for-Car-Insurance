"""Writes thin, display-only notebooks (one per stage) into notebooks/. They read the
tables and figures produced by the pipeline; all computation lives in src/. Generated and
executed by `make notebooks` (Jupyter is in requirements-dev.txt, not requirements.txt)."""
import json

from src.config import ROOT

HEADER = """import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.config import load_config

cfg = load_config()
T, F = cfg["paths"]["tables"], cfg["paths"]["figures"]
pd.set_option("display.width", 200)


def show(name):
    plt.figure(figsize=(12, 5))
    plt.imshow(mpimg.imread(F / name))
    plt.axis("off")
    plt.show()"""

STAGES = [
    ("01_data_and_cleaning", "Stage 1: data and cleaning (DECISIONS D001-D012)",
     ["data_source", "cleaning_log", "data_summary", "split_balance", "severity_tail_thresholds"], ["severity_tail.png"]),
    ("02_banding", "Stage 2: banding and features (D013-D017)",
     ["band_exposure_check", "area_vs_density", "band_vs_spline_cv"], ["oneway_DrivAge_band.png", "oneway_BonusMalus_band.png"]),
    ("03_frequency_models", "Stage 3: frequency models (D019-D022)",
     ["frequency_models_stage3", "gbm_chosen", "log_exposure_diagnostic", "exposure_by_segment"],
     ["lift_glm_a_gbm_holdout.png", "double_lift_gbm_vs_glm_a_holdout.png", "ae_by_factor_holdout_stage3.png"]),
    ("04_interactions", "Stage 4: SHAP interactions and GLM-B (D018, D023-D030, D039)",
     ["shap_interaction_ranking", "two_way_ae_summary_glm_a", "interaction_selection_log", "interaction_accepted",
      "gap_closed", "frequency_model_comparison", "bm_monotonicity_steps"],
     ["ae_heatmap_DrivAge_x_BonusMalus.png", "shap_dependence_DrivAge_x_BonusMalus.png", "lift_holdout.png"]),
    ("05_severity_burning_cost", "Stage 5: severity and burning cost (D031-D033)",
     ["severity_selection_log", "severity_model_comparison", "severity_residual_summary", "burning_cost_reconciliation",
      "large_loss_sensitivity"], ["severity_residuals_learn_oof.png"]),
    ("06_tariff", "Stage 6: tariff (D034-D039)",
     ["tariff_summary", "relativities_DrivAge_band", "relativities_BonusMalus_bandB", "tariff_vs_gbm_summary",
      "tariff_exposure_control_premium_ratio", "glm_a_factor_importance", "bm_crm_sensitivity"],
     ["double_lift_tariff_vs_gbm_holdout.png"]),
    ("07_impact", "Stage 7: impact analysis (D040-D041)",
     ["impact_bridge", "impact_change_bands", "impact_top_segments", "impact_capping"], ["impact_change_histogram.png"]),
]


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def write() -> list:
    out = []
    d = ROOT / "notebooks"
    d.mkdir(exist_ok=True)
    for name, title, tables, figs in STAGES:
        cells = [cell("markdown", f"# {title}\n\nDisplay only. Run `make all` first; this notebook reads `reports/`."),
                 cell("code", "import os, sys\nsys.path.insert(0, os.path.abspath('..'))\n" + HEADER)]
        cells += [cell("code", f'pd.read_csv(T / "{t}.csv")') for t in tables]
        cells += [cell("code", f'show("{f}")') for f in figs]
        nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        p = d / f"{name}.ipynb"
        p.write_text(json.dumps(nb, indent=1) + "\n")
        out.append(p)
    return out


if __name__ == "__main__":
    for p in write():
        print(p.relative_to(ROOT))
