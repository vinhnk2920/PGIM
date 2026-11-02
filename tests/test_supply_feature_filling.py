"""Preserve supply geography and project identity during forecast preparation."""

import numpy as np
import pandas as pd
import pytest

from functions.config import Settings
from functions.forecasting.data_loader import fill_missing
from functions.forecasting import prepare_rental, prepare_sales


def panel():
    dates = pd.date_range("2018-01-01", "2020-12-01", freq="MS")
    parts = []
    for project_idx, name, district, tier in (
        (0, "Project A", 9, "SZ1"),
        (1, "Project A", 9, "SZ2"),
        (2, "Project B", 15, "SZ1"),
    ):
        i = np.arange(len(dates))
        price = 3.0 + project_idx + i / 50
        observed = (i + project_idx) % 6 != 0
        frame = pd.DataFrame({
            "project_idx": project_idx, "condo_name": name, "district": district,
            "size_tier": tier, "date": dates, "period": dates.year * 100 + dates.month,
            "month_of_year": dates.month, "Planning Area": "Central" if district == 9 else "East",
            "condo_age_at_transaction": 10.0, "tenure_remaining_years": 80.0,
            "nearest_mrt_dist": 300.0, "Large_Dev_200plus": 0,
            "was_observed": observed, "rent_psf_imp": price,
            "rent_psf_obs": np.where(observed, price, np.nan),
            "project_sold_share_lag1m": 0.2 + i / 1000 if district == 9 else 0.9 + i / 1000,
            "district_units_launched_lag1m": 10.0 + i if district == 9 else 90.0 + i,
            "district_units_under_construction_lag1q": 100.0 if district == 9 else 900.0,
            "rental_index_nonlanded_yoy_log_growth_lag1q": i / 100,
            "cpi_all_items_infl_yoy_pct": i.astype(float),
        })
        for radius in (500, 1000):
            for suffix in (
                "median_psf_6m_lag1m", "median_psf_12m_lag1m", "observed_condos_12m_lag1m",
                "known_condos_lag1m", "median_psf_6m_log_growth_12m_lag1m",
            ):
                frame[f"nearby_{radius}m_{suffix}"] = 1.0
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


@pytest.mark.parametrize("market", ["rental", "sales"])
def test_real_forecast_preparers_preserve_complete_supply_values(tmp_path, market):
    settings = Settings(source_root=tmp_path / "data", output_root=tmp_path / "outputs")
    source = panel()
    expected = source.set_index(["project_idx", "date"])[[
        "project_sold_share_lag1m", "district_units_launched_lag1m",
        "district_units_under_construction_lag1q",
    ]]
    if market == "sales":
        source = source.rename(columns={"rent_psf_obs": "sale_psf_obs", "rent_psf_imp": "sale_psf_imp"})
    path = settings.model_input(market)
    path.parent.mkdir(parents=True)
    source.to_csv(path, index=False)
    buckets = settings.bucket_ranges()
    buckets.parent.mkdir(parents=True)
    buckets.write_text("size_bucket,lower_bound,upper_bound\nSZ1,0,650\nSZ2,650,2000\n", encoding="utf-8")
    module = prepare_rental if market == "rental" else prepare_sales
    prepared, lists, _, _ = module.prepare(settings)
    actual = prepared.set_index(["project_idx", "date"])[expected.columns].sort_index()
    np.testing.assert_allclose(actual.to_numpy(), expected.sort_index().to_numpy(), rtol=1e-6)
    for key in ("FEATURES_A", "FEATURES_OBS", "FEATURES_T12", "FEATURES_T24"):
        assert {"project_sold_share_lag1m", "district_units_launched_lag1m"} <= set(lists[key])


def test_missing_supply_uses_same_entity_month_or_earlier_month():
    source = panel()
    source.loc[(source.project_idx == 0) & (source.period == 201802), "project_sold_share_lag1m"] = np.nan
    source.loc[(source.condo_name == "Project A") & (source.period == 201803), "project_sold_share_lag1m"] = np.nan
    source.loc[(source.district == 9) & (source.period == 201803), "district_units_launched_lag1m"] = np.nan
    result = fill_missing(source.copy(), 202001)
    # Another size series supplies the same project/month; otherwise only past history is used.
    assert result.loc[(result.project_idx == 0) & (result.period == 201802), "project_sold_share_lag1m"].item() == pytest.approx(0.201)
    assert result.loc[(result.project_idx == 0) & (result.period == 201803), "project_sold_share_lag1m"].item() == pytest.approx(0.201)
    assert result.loc[(result.project_idx == 0) & (result.period == 201803), "district_units_launched_lag1m"].item() == 11
    known = source.project_sold_share_lag1m.notna()
    pd.testing.assert_series_equal(result.loc[known, "project_sold_share_lag1m"], source.loc[known, "project_sold_share_lag1m"])


def test_filling_preserves_row_order_index_and_protected_prices():
    source = panel().sample(frac=1, random_state=4)
    source.index = np.arange(len(source)) * 3 + 7
    result = fill_missing(source.copy(), 202001)
    columns = ["project_idx", "condo_name", "district", "date", "period", "rent_psf_obs", "rent_psf_imp", "was_observed"]
    pd.testing.assert_frame_equal(result[columns], source[columns])


def test_future_changes_do_not_fill_leading_unknown_supply_or_other_projects():
    source = panel()
    source.loc[(source.condo_name == "Project A") & (source.period <= 201803), "project_sold_share_lag1m"] = np.nan
    source.loc[(source.district == 9) & (source.period <= 201803), "district_units_launched_lag1m"] = np.nan
    source.loc[source.condo_name == "Project B", "project_sold_share_lag1m"] = np.nan
    baseline = fill_missing(source.copy(), 202001)
    changed = source.copy()
    changed.loc[(changed.condo_name == "Project A") & (changed.period >= 202001), "project_sold_share_lag1m"] = 1.0
    changed.loc[(changed.district == 9) & (changed.period >= 202001), "district_units_launched_lag1m"] = 5000.0
    result = fill_missing(changed, 202001)
    columns = ["project_sold_share_lag1m", "district_units_launched_lag1m"]
    pd.testing.assert_frame_equal(result.loc[result.period < 202001, columns], baseline.loc[baseline.period < 202001, columns])
    assert baseline.loc[(baseline.condo_name == "Project A") & (baseline.period <= 201803), "project_sold_share_lag1m"].isna().all()
    assert baseline.loc[baseline.condo_name == "Project B", "project_sold_share_lag1m"].isna().all()
    assert baseline.loc[(baseline.district == 9) & (baseline.period <= 201803), "district_units_launched_lag1m"].isna().all()


def test_conflicting_values_for_same_entity_month_raise():
    source = panel()
    source.loc[(source.project_idx == 1) & (source.period == 201801), "project_sold_share_lag1m"] = 0.8
    with pytest.raises(ValueError, match="Conflicting.*project_sold_share_lag1m"):
        fill_missing(source, 202001)


def test_national_macro_lag_keeps_calendar_policy():
    source = panel()
    missing = (source.project_idx == 0) & (source.period == 201805)
    source.loc[missing, "rental_index_nonlanded_yoy_log_growth_lag1q"] = np.nan
    result = fill_missing(source.copy(), 202001)
    assert result.loc[missing, "rental_index_nonlanded_yoy_log_growth_lag1q"].item() == pytest.approx(0.04)
    assert result.loc[(result.project_idx == 0) & (result.period == 201807), "cpi_all_items_infl_yoy_pct"].item() == 3


def test_lightgbm_accepts_unknown_initial_supply():
    from lightgbm import LGBMRegressor

    source = panel()
    source.loc[source.period <= 201806, "project_sold_share_lag1m"] = np.nan
    prepared = fill_missing(source, 202001)
    features = ["project_sold_share_lag1m", "district_units_launched_lag1m"]
    model = LGBMRegressor(n_estimators=4, num_leaves=4, min_child_samples=2, n_jobs=1, verbosity=-1)
    model.fit(prepared[features], prepared["rent_psf_imp"])
    assert np.isfinite(model.predict(prepared[features])).all()
