import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.gbm import FrequencyRankEncoder, features


def test_encoder_uses_training_rows_only():
    df = pd.DataFrame({"VehBrand": ["a", "a", "b", "b", "c", "c"], "ClaimNb": [0, 1, 1, 1, 0, 0],
                       "Exposure": [1.0] * 6})
    train, valid = df.iloc[:4], df.iloc[4:]
    enc = FrequencyRankEncoder(["VehBrand"]).fit(train)
    before = dict(enc.maps["VehBrand"])
    valid = valid.assign(ClaimNb=[4, 4])  # changing validation outcomes cannot move the encoding
    enc2 = FrequencyRankEncoder(["VehBrand"]).fit(train)
    assert enc2.maps["VehBrand"] == before
    assert "c" not in before  # level only in validation is not ranked from validation data
    assert enc.transform(valid)["VehBrand"].notna().all()


def test_tuning_encoders_never_fitted_on_score_or_early_stopping_fold(stage3):
    for trial in stage3["gbm_tuning_meta"]:
        for m in trial["fold_meta"]:
            fitted = set(m["encoder_fitted_on_folds"])
            assert m["score_fold"] not in fitted
            assert m["early_stopping_fold"] not in fitted
            assert -1 not in fitted  # holdout
            assert len(fitted) == 3


def test_final_encoder_fitted_on_full_learn_set_only(stage3, banded):
    learn, hold = banded
    fit_index = set(stage3["gbm_encoder"].fit_index)
    assert fit_index == set(learn.index)
    assert fit_index.isdisjoint(hold.index)


def test_gbm_monotone_in_bonusmalus(stage3, banded, cfg):
    learn, _ = banded
    model = lgb.Booster(model_str=stage3["gbm_model"])
    rows = learn.sample(200, random_state=0)
    grid = np.arange(50, 231, 5)
    X = features(rows, cfg, stage3["gbm_encoder"])
    preds = []
    for bm in grid:
        preds.append(model.predict(X.assign(BonusMalus=float(bm))))
    preds = np.array(preds)
    assert (np.diff(preds, axis=0) >= -1e-12).all()


def test_frequency_glms_balance_on_learn(stage3, banded):
    learn, _ = banded
    for name in ("glm_0", "glm_a", "glm_b"):
        if name not in stage3:
            continue
        pred = stage3[name].predict(learn, np.log(learn["Exposure"].to_numpy())).sum()
        assert abs(pred / learn["ClaimNb"].sum() - 1) < 0.005, name
