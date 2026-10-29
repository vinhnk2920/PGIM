"""Meaningful safeguards for calendar joins, train boundaries and shared helpers."""

import importlib

import numpy as np
import pandas as pd
import pytest

from functions.common import add_months, rolling_windows
from functions.config import Settings
from functions.forecasting.pipeline import (
    cv_folds,
    horizon_frame,
    training_rows,
    fit_model,
    metric_row,
)
from functions.preprocessing.pipeline import area_midpoint, remove_outliers


def panel():
    dates = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
    parts = []
    for project in range(4):
        i = np.arange(len(dates))
        observed = (i + project) % 7 != 0
        price = 1000 + project * 150 + i * 3 + np.sin(i / 12) * 50
        parts.append(
            pd.DataFrame(
                dict(
                    project_idx=project,
                    period=dates.year * 100 + dates.month,
                    date=dates,
                    rent_psf_imp=price,
                    rent_psf_obs=np.where(observed, price, np.nan),
                    was_observed=observed,
                    feature=np.sin(i / 12),
                )
            )
        )
    return pd.concat(parts, ignore_index=True)


@pytest.mark.parametrize(
    "period,shift,expected",
    [(202012, 1, 202101), (202001, -1, 201912), (202402, 24, 202602), (202001, -25, 201712)],
)
def test_calendar_rollover(period, shift, expected):
    assert add_months(period, shift) == expected


def test_rolling_windows_cover_each_origin_once():
    windows = rolling_windows(202001, 202205, 180, 12)
    assert windows[0]["train_start"] == 200501
    assert windows[-1]["test_end"] == 202205
    origins = []
    for w in windows:
        origin = w["test_start"]
        while origin <= w["test_end"]:
            origins.append(origin)
            origin = add_months(origin, 1)
    assert len(origins) == 29 == len(set(origins))


@pytest.mark.parametrize("horizon", [1, 12, 24])
def test_training_targets_strictly_before_test_and_validation(horizon):
    frame = horizon_frame(panel(), horizon, target_mode="log_growth")
    window = dict(train_start=200001, train_end=201912, test_start=202001)
    train = frame.loc[training_rows(frame, window, horizon, 200001)]
    assert train["target_period"].max() < 202001
    settings = Settings()
    for x_train, _, x_valid, _ in cv_folds(train, ["feature"], horizon, settings):
        assert (
            train.loc[x_train.index, "target_period"].max()
            < train.loc[x_valid.index, "period"].min()
        )


def test_horizon_uses_calendar_month_not_next_available_row():
    frame = panel()
    frame = frame.loc[frame.period.ne(200002)]
    result = horizon_frame(frame, 1)
    assert result.loc[result.period.eq(200001), "target_level"].isna().all()


def test_observed_evaluation_and_unknown_future():
    frame = horizon_frame(panel(), 24, observed_only=True)
    assert frame.was_observed.all() and frame.target_observed.all()
    future = horizon_frame(panel(), 24)
    assert future.loc[future.period.eq(202512), "target_level"].isna().all()
    assert future.fair_eval_row.equals(future.was_observed & future.target_observed)


def test_log_growth_reconstructs_price_level():
    frame = horizon_frame(panel(), 12, target_mode="log_growth").dropna(subset=["model_target"])
    np.testing.assert_allclose(np.exp(frame.base_log + frame.model_target), frame.target_level)


@pytest.mark.parametrize("market", ["sales", "rental"])
def test_weighted_lightgbm_smoke(market, tmp_path):
    settings = Settings(output_root=tmp_path, fast=True, horizons=(1,))
    frame = horizon_frame(panel(), 1, target_mode="log_growth" if market == "sales" else "level")
    train = frame.loc[frame.period.lt(202001) & frame.model_target.notna()]
    _, model, study = fit_model(train, ["feature"], [], market, "imputed", 1, settings)
    predicted = model.predict(frame[["feature"]])
    assert len(predicted) == len(frame) and np.isfinite(predicted).all() and study is None


def test_metrics_ignore_unobserved_rows():
    frame = pd.DataFrame(
        dict(
            target_level=[10.0, 20.0, 10000.0],
            prediction=[11.0, 18.0, 0.0],
            fair_eval_row=[True, True, False],
        )
    )
    row = metric_row(frame, "prediction", "rental", 1, 1, "imputed")
    assert row["n"] == 2 and row["MAE"] == 1.5


def test_area_parsing_and_shared_outlier_rule():
    assert area_midpoint("1,000 to 1,200") == 1100
    assert area_midpoint("900") == 900
    df = pd.DataFrame(
        {
            "Property Type": ["Non-landed Properties"] * 100,
            "Floor Area (SQFT)": ["1000 to 1200"] * 100,
            "Monthly Rent ($)": [str(x) for x in range(1000, 1100)],
        }
    )
    cleaned, audit = remove_outliers(df, "rental")
    assert len(cleaned) == 98 and audit["removed_rows"] == 2


def test_imports_do_not_start_training_or_create_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for module in [
        "functions.features.supply",
        "functions.features.labour",
        "functions.features.consulting",
        "functions.features.trend_policy",
        "functions.features.nearby_prices",
        "functions.imputation.rental",
        "functions.imputation.sales",
        "functions.forecasting.prepare_rental",
        "functions.forecasting.prepare_sales",
    ]:
        importlib.import_module(module)
    assert not list(tmp_path.iterdir())


def test_invalid_config_rejected(tmp_path):
    with pytest.raises(ValueError):
        Settings(output_root=tmp_path, horizons=(3,))
    with pytest.raises(ValueError):
        Settings(output_root=tmp_path, imputed_target_weight=2)


def test_default_inputs_are_inside_project_data():
    from functions.config import DEFAULT_SOURCE, PACKAGE_ROOT

    assert DEFAULT_SOURCE == PACKAGE_ROOT / "data"


def test_retained_reporting_module_imports_and_calendar_boundary():
    from functions.forecasting import eval_reporting
    from functions.forecasting.rolling_helpers import effective_origin_train_end

    assert callable(eval_reporting.fair_subset)
    assert effective_origin_train_end(202001, 12) == 201812


def test_local_canonical_imputation_and_forecast_checkpoints(tmp_path):
    settings = Settings(
        source_root=tmp_path / "data", output_root=tmp_path / "outputs", reuse_existing=True
    )
    imputed = settings.source_root / "imputation" / settings.imputed("sales").name
    imputed.parent.mkdir(parents=True)
    imputed.write_text("sample", encoding="utf-8")
    model_input = settings.source_root / "model_input" / settings.model_input("sales").name
    assert settings.read_imputed("sales") == imputed
    assert settings.read_model_input("sales") == model_input


@pytest.mark.parametrize("market", ["sales", "rental"])
def test_weighted_tuning_scores_observed_rows_in_price_units(market):
    from functions.forecasting.tuning import (
        run_optuna_log_growth_level_rmse,
        run_optuna_target_weighted,
    )

    settings = Settings()
    frame = horizon_frame(panel(), 1, target_mode="log_growth" if market == "sales" else "level")
    frame = frame.loc[frame["period"].lt(202001) & frame.model_target.notna()]
    folds = cv_folds(frame, ["feature"], 1, settings)
    weights = pd.Series(np.where(frame.target_observed, 1.0, 0.0), index=frame.index)
    kwargs = dict(
        n_trials=1, fast_mode=False, n_estimators_max_search=20, early_stop=3, n_cv_folds_search=2
    )
    if market == "sales":
        _, model, study = run_optuna_log_growth_level_rmse(
            frame[["feature"]],
            frame.model_target,
            folds,
            [],
            level_context_df=frame,
            base_log_col="base_log",
            level_target_col="target_level",
            sample_weight_context=weights,
            eval_mask_col="fair_eval_row",
            **kwargs,
        )
    else:
        _, model, study = run_optuna_target_weighted(
            frame[["feature"]],
            frame.model_target,
            folds,
            [],
            sample_weight_context=weights,
            eval_context_df=frame,
            eval_mask_col="fair_eval_row",
            **kwargs,
        )
    assert len(study.trials) == 1
    assert all(n > 0 for n in study.best_trial.user_attrs["fair_eval_rows"])
    assert np.isfinite(model.predict(frame[["feature"]])).all()
