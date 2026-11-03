"""September repair preserves the existing rental lag and observed input data."""

import numpy as np
import pandas as pd
import pytest

from functions.preprocessing.monthly import (
    RENTAL_MONTHLY_MACRO_COLS, fill_rental_monthly_macro_gaps,
)


def reference(tmp_path):
    path = tmp_path / "macro.csv"
    pd.DataFrame({"month": ["2019-07-01", "2019-08-01", "2019-09-01", "2019-10-01"],
                  "cpi_all_items_infl_yoy_pct": [1., 2., 100., 200.],
                  "sora_compounded_3m_pct": [.5, .7, 8., 9.]}).to_csv(path, index=False)
    return path


def test_missing_september_uses_august_without_replacing_known_months(tmp_path):
    frame = pd.DataFrame({"date": ["2019-09", "2019-08", "2019-10", "2019-09"],
                          "cpi_all_items_infl_yoy_pct": [np.nan, 1., 99., 2.5],
                          "sora_compounded_3m_pct": [np.nan, .5, 7., .75],
                          "rent_psf_obs": [3., np.nan, 4., 5.],
                          "rent_psf_imp": [3., 3.5, 4., 5.],
                          "was_observed": [True, False, True, True]}, index=[7, 2, 5, 9])
    before = frame.copy(deep=True)
    result = fill_rental_monthly_macro_gaps(frame, "date", reference(tmp_path))
    assert result.loc[7, "cpi_all_items_infl_yoy_pct"] == 2.
    assert result.loc[7, "sora_compounded_3m_pct"] == .7
    for column in ("cpi_all_items_infl_yoy_pct", "sora_compounded_3m_pct"):
        known = before[column].notna()
        pd.testing.assert_series_equal(result.loc[known, column], before.loc[known, column])
    other = [name for name in before.columns if name not in RENTAL_MONTHLY_MACRO_COLS]
    pd.testing.assert_frame_equal(result[other], before[other])
    pd.testing.assert_frame_equal(frame, before)


def test_future_reference_changes_do_not_change_past_repair(tmp_path):
    path = reference(tmp_path)
    frame = pd.DataFrame({"date": ["2019-09"], "cpi_all_items_infl_yoy_pct": [np.nan]})
    before = fill_rental_monthly_macro_gaps(frame, "date", path)
    source = pd.read_csv(path)
    source.loc[source["month"].ge("2019-09-01"), "cpi_all_items_infl_yoy_pct"] = 1e6
    source.to_csv(path, index=False)
    pd.testing.assert_frame_equal(fill_rental_monthly_macro_gaps(frame, "date", path), before)


def test_missing_source_value_stays_unknown_and_does_not_use_later_month(tmp_path):
    path = reference(tmp_path)
    source = pd.read_csv(path)
    source.loc[source["month"].eq("2019-08-01"), "cpi_all_items_infl_yoy_pct"] = np.nan
    source.to_csv(path, index=False)
    frame = pd.DataFrame({"date": ["2019-09", "2010-01"], "cpi_all_items_infl_yoy_pct": [np.nan, np.nan]})
    assert fill_rental_monthly_macro_gaps(frame, "date", path)["cpi_all_items_infl_yoy_pct"].isna().all()


def test_duplicate_reference_months_are_rejected(tmp_path):
    path = reference(tmp_path)
    source = pd.read_csv(path)
    pd.concat([source, source.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        fill_rental_monthly_macro_gaps(
            pd.DataFrame({"date": ["2019-09"], "cpi_all_items_infl_yoy_pct": [np.nan]}),
            "date", path,
        )


def test_price_only_inputs_do_not_need_a_macro_reference(tmp_path):
    frame = pd.DataFrame({"date": ["2019-09"], "rent_psf_imp": [3.]})
    result = fill_rental_monthly_macro_gaps(frame, "date", tmp_path / "absent.csv", copy=False)
    assert result is frame


def test_repaired_september_reaches_december_after_existing_three_month_shift(tmp_path):
    from functions.forecasting.data_loader import fill_missing

    dates = pd.date_range("2019-01-01", "2019-12-01", freq="MS")
    frame = pd.DataFrame({"date": dates, "period": dates.year * 100 + dates.month,
                          "condo_name": "A", "district": 9, "nearest_mrt_dist": 300.,
                          "condo_age_at_transaction": 10., "tenure_remaining_years": 80.,
                          "Large_Dev_200plus": 0, "cpi_all_items_infl_yoy_pct": 1.})
    frame.loc[frame.date.eq("2019-09-01"), "cpi_all_items_infl_yoy_pct"] = np.nan
    repaired = fill_rental_monthly_macro_gaps(frame, "date", reference(tmp_path))
    result = fill_missing(repaired, 202001)
    assert result.loc[result.period.eq(201912), "cpi_all_items_infl_yoy_pct"].item() == 2.


def test_repair_columns_match_existing_core_forecast_macros():
    from functions.forecasting.data_loader import MACRO_COLS_CORE

    assert set(RENTAL_MONTHLY_MACRO_COLS) == set(MACRO_COLS_CORE)


def test_preprocessing_repairs_metadata_before_export(tmp_path, monkeypatch):
    from functions.config import Settings
    from functions.preprocessing import pipeline

    settings = Settings(source_root=tmp_path / "data", output_root=tmp_path / "out")
    source = settings.transaction_input("rental", "enriched")
    source.parent.mkdir(parents=True)
    frame = pd.DataFrame({"Lease Commencement Date": ["2019-08-01", "2019-09-01", "2019-10-01"],
                          "cpi_all_items_infl_yoy_pct": [1., np.nan, 99.],
                          "Monthly Rent ($)": [3000., 3200., 3100.]})
    frame.to_csv(source, index=False)
    macro_path = settings.prep_path("macro", "monthly", "sg_macro_monthly.csv")
    macro_path.parent.mkdir(parents=True)
    macro_path.write_bytes(reference(tmp_path).read_bytes())
    monkeypatch.setattr(pipeline, "assessed_reference", lambda *args: None)
    monkeypatch.setattr(pipeline, "add_quarterly_macros", lambda df, *args, **kwargs: df)
    monkeypatch.setattr(pipeline, "remove_outliers", lambda df, market: (df, {}))
    pipeline.run(settings, markets=("rental",))
    output = pd.read_csv(settings.prepared("rental"))
    assert output["cpi_all_items_infl_yoy_pct"].tolist() == [1., 2., 99.]
    other = [name for name in frame if name != "cpi_all_items_infl_yoy_pct"]
    pd.testing.assert_frame_equal(output[other], frame[other])
