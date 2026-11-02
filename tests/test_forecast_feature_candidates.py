"""Keep declared candidates aligned with the agreed existing-model feature set."""

import pandas as pd

from functions.forecasting.data_loader import MACRO_COLS, SCOPED_SUPPLY_COLS
from functions.forecasting.feature_engineering import BASE_STRUCTURAL, make_feature_lists


RETIRED = {
    "project_sold_share_mom_chg_lag1m",
    "district_units_under_construction_qoq_lag1q",
    "district_units_not_launched_yet_qoq_lag1q",
    "district_units_launched_unsold_qoq_lag1q",
    "district_units_completed_unsold_qoq_lag1q",
    "labour_churn_index_lag1q",
    "ep_share_yoy_pct_lag1y",
}
RETAINED = {
    "project_sold_share_lag1m", "district_units_launched_lag1m",
    "retrenchment_yoy_pct_lag1q", "median_income_yoy_pct_lag",
}


def feature_lists(frame, structural=None):
    return make_feature_lists(
        frame, [], {"h24_noimp_core_macros_only": False},
        structural_features=structural,
        lag1_primary=[], lag1_secondary=[], lag12_primary=[], lag12_secondary=[],
        lag24_primary=[], lag24_secondary=[],
    )


def test_retired_unimplemented_names_are_not_default_candidates_or_fill_groups():
    assert RETIRED.isdisjoint(MACRO_COLS)
    assert RETIRED.isdisjoint(BASE_STRUCTURAL)
    assert RETIRED.isdisjoint(SCOPED_SUPPLY_COLS)
    assert RETAINED <= set(BASE_STRUCTURAL)


def test_legacy_named_columns_cannot_silently_reenter_default_models(capsys):
    frame = pd.DataFrame(columns=list(BASE_STRUCTURAL) + sorted(RETIRED))
    before = frame.copy(deep=True)
    lists = feature_lists(frame)
    for key, names in lists.items():
        if key.startswith("FEATURES_"):
            assert RETIRED.isdisjoint(names)
            assert RETAINED <= set(names)
    pd.testing.assert_frame_equal(frame, before)
    assert "unavailable candidates skipped" not in capsys.readouterr().out


def test_missing_candidates_are_reported_once_without_adding_or_renaming_columns(capsys):
    frame = pd.DataFrame({"available": [1.0]})
    lists = feature_lists(frame, ["available", "missing_candidate", "missing_candidate"])
    assert lists["FEATURES_A"] == ["available"]
    output = capsys.readouterr().out
    assert "WARNING: 1 unavailable candidates skipped: missing_candidate" in output
    assert output.count("missing_candidate") == 1
    assert frame.columns.tolist() == ["available"]
