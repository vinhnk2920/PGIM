"""Reproduction must refit independently and preserve scoring keys and inputs."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from functions.cli import parser
from functions.config import Settings
from functions.forecasting.reproduce import run_reference, target_arrays, fingerprint


def test_reported_weight_is_default_but_w0_remains_explicit():
    assert Settings().imputed_target_weight == 1
    assert parser("forecast").parse_args([]).imputed_target_weight == 1
    assert parser("forecast").parse_args(["--imputed-target-weight", "0"]).imputed_target_weight == 0


def test_calendar_gap_does_not_become_next_available_target():
    panel = pd.DataFrame({"condo_name": ["A"] * 3, "size_tier": ["SZ1"] * 3,
                          "period": [201912, 202002, 202003], "rent_psf_imp": [100., 110., 120.],
                          "was_observed": [True, False, True]})
    _, levels, observed, _ = target_arrays(panel, 1, "level")
    assert np.isnan(levels[0]) and levels[1] == 120 and observed[1]
    with pytest.raises(ValueError, match="duplicate"):
        target_arrays(pd.concat([panel, panel.iloc[:1]], ignore_index=True), 1, "level")


def setup_reference(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    dates = pd.date_range("2000-01-01", "2004-12-01", freq="MS")
    panel = pd.DataFrame({"condo_name": "A", "size_tier": "SZ1",
                          "period": dates.year * 100 + dates.month,
                          "rent_psf_imp": 100 + np.arange(len(dates), dtype=float),
                          "was_observed": True, "feature": np.arange(len(dates), dtype=float) % 7})
    target, levels, _, periods = target_arrays(panel, 1, "level")
    train = (panel.period >= 200001) & (periods < 200301) & np.isfinite(target)
    model = LGBMRegressor(n_estimators=3, num_leaves=3, min_child_samples=5, random_state=42,
                          deterministic=True, force_col_wise=True, n_jobs=1, verbose=-1)
    model.fit(panel.loc[train, ["feature"]], target[train], sample_weight=np.ones(train.sum()))
    joblib.dump(dict(model=model, features=["feature"], cat_features=[], target_mode="level"), archive / "rental_h1.pkl")
    panel.rename(columns={"period": "yyyymm"}).to_csv(archive / "features.csv", index=False)
    test = panel.period.ge(200301) & np.isfinite(levels)
    pred = panel.loc[test, ["condo_name", "size_tier", "period"]].copy()
    pred["horizon"], pred["fair_eval_row"], pred["y_true"] = 1, True, levels[test]
    pred["y_pred_imputed"] = model.predict(panel.loc[test, ["feature"]])
    pred.to_csv(archive / "predictions.csv", index=False)
    profile = {"schema_version": 1, "name": "test", "train_start": 200001,
               "test_cutoff": 200301, "imputed_target_weight": 1., "numeric_dtype": "float64",
               "error_metrics_rtol": 1e-6, "r2_atol": 1e-6,
               "markets": {"rental": {"reference_run": str(archive), "feature_panel": "features.csv",
                                       "model_pattern": "rental_h{horizon}.pkl",
                                       "reference_predictions": "predictions.csv"}}}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return archive, path


@pytest.mark.parametrize("threads", [None, 4])
def test_fresh_refit_matches_and_keeps_reference_files_unchanged(tmp_path, threads):
    archive, profile = setup_reference(tmp_path)
    before = {p.name: fingerprint(p)["sha256"] for p in archive.iterdir()}
    settings = Settings(reference_profile=profile, output_root=tmp_path / "out",
                        horizons=(1,), test_cutoff=200301, reference_threads=threads)
    result = run_reference(settings, ("rental",))
    comparison = pd.read_csv(result / "comparison.csv")
    assert comparison.passed.all()
    metrics = pd.read_csv(result / "metrics.csv")
    assert set(metrics.model) == {"reference", "archived_model_inference", "fresh_refit"}
    assert (result / "rental_h1_refit.pkl").exists()
    assert before == {p.name: fingerprint(p)["sha256"] for p in archive.iterdir()}


def test_conflicting_weight_fails_instead_of_silently_changing_reference(tmp_path):
    _, profile = setup_reference(tmp_path)
    settings = Settings(reference_profile=profile, output_root=tmp_path / "out", horizons=(1,),
                        test_cutoff=200301, imputed_target_weight=0)
    with pytest.raises(ValueError, match="imputed_target_weight"):
        run_reference(settings, ("rental",))


def test_outputs_cannot_be_written_inside_archive(tmp_path):
    archive, profile = setup_reference(tmp_path)
    settings = Settings(reference_profile=profile, output_root=archive / "out", horizons=(1,),
                        test_cutoff=200301)
    with pytest.raises(ValueError, match="inside the reference"):
        run_reference(settings, ("rental",))
    assert not (archive / "out").exists()
