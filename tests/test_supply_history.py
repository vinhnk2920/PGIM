"""Project supply must depend only on observations through the previous month."""

from types import SimpleNamespace

import pandas as pd
import pytest

from functions.features import supply


def project_history():
    return pd.DataFrame({
        "condo_name_key": ["A", "A", "A", "A", "B", "B"],
        "date": pd.to_datetime([
            "2020-01-01", "2020-02-01", "2020-04-01", "2020-05-01",
            "2020-01-01", "2020-04-01",
        ]),
        "cumul_sold": [20, 80, None, 100, None, 30],
        "total_units": [100] * 6,
    }, index=[9, 4, 15, 7, 30, 2])


def indexed_features(frame):
    return supply.build_project_sold_shares(frame).set_index(["condo_name_key", "date"]).sort_index()


def test_initial_lag_stays_unknown_even_if_project_eventually_sells_out():
    frame = project_history()
    before = frame.copy(deep=True)
    result = indexed_features(frame)
    assert pd.isna(result.loc[("A", "2020-01-01"), "project_sold_share_lag1m"])
    assert result.loc[("A", "2020-02-01"), "project_sold_share_lag1m"] == 0.2
    assert pd.isna(result.loc[("B", "2020-04-01"), "project_sold_share_lag1m"])
    assert result.loc[("A", "2020-05-01"), "project_sold_share_lag1m"] == 0.8
    pd.testing.assert_frame_equal(frame, before)


def test_missing_calendar_month_uses_latest_past_share():
    frame = project_history().iloc[[0, 2, 3]].copy()
    result = indexed_features(frame)
    for month in ("2020-02-01", "2020-03-01", "2020-04-01"):
        assert result.loc[("A", month), "project_sold_share_lag1m"] == 0.2
    assert result.loc[("A", "2020-04-01"), "project_sold_share"] == 0.2
    frame.loc[7, "cumul_sold"] = None
    assert indexed_features(frame).loc[("A", "2020-05-01"), "project_sold_share_lag1m"] == 0.2


@pytest.mark.parametrize("future_share", [80, 100])
def test_future_perturbation_and_extension_do_not_change_earlier_features(future_share):
    earlier = project_history().loc[[9, 4, 30]]
    extended = pd.concat([
        earlier,
        pd.DataFrame({"condo_name_key": ["A", "B"],
                      "date": pd.to_datetime(["2020-12-01", "2020-12-01"]),
                      "cumul_sold": [future_share, 100], "total_units": [100, 100]},
                     index=[50, 51]),
    ])
    expected = indexed_features(earlier)
    actual = indexed_features(extended).reindex(expected.index)
    pd.testing.assert_frame_equal(actual, expected)


def test_row_order_and_entity_isolation():
    frame = project_history()
    expected = indexed_features(frame)
    shuffled = frame.sample(frac=1, random_state=42)
    pd.testing.assert_frame_equal(indexed_features(shuffled), expected)
    assert expected.loc[("B", "2020-04-01"), "project_sold_share"] == 0.3
    assert pd.isna(expected.loc[("B", "2020-04-01"), "project_sold_share_lag1m"])


def test_last_report_is_available_only_from_the_next_month():
    result = indexed_features(project_history())
    assert result.loc[("A", "2020-05-01"), "project_sold_share_lag1m"] == 0.8
    assert result.loc[("A", "2020-06-01"), "project_sold_share_lag1m"] == 1.0
    assert result.loc[("B", "2020-06-01"), "project_sold_share_lag1m"] == 0.3


def test_current_report_cannot_change_the_same_month_lag():
    frame = project_history()
    before = indexed_features(frame)
    frame.loc[4, "cumul_sold"] = 100
    after = indexed_features(frame)
    for month in ("2020-01-01", "2020-02-01"):
        pd.testing.assert_series_equal(after.loc[("A", month), ["project_sold_share_lag1m"]],
                                       before.loc[("A", month), ["project_sold_share_lag1m"]])
    assert after.loc[("A", "2020-03-01"), "project_sold_share_lag1m"] == 1.0


def test_duplicate_project_month_is_rejected():
    frame = project_history().iloc[[0, 0]]
    with pytest.raises(ValueError, match="Duplicate project/month"):
        supply.build_project_sold_shares(frame)


def test_actual_supply_stage_preserves_unknown_lags_for_both_markets(tmp_path, monkeypatch):
    districts = sorted(set(range(1, 29)) - {24})
    panel = pd.DataFrame([
        {"project_idx": district, "condo_name": f"PROJECT {district}",
         "district": district, "date": date, "sentinel": district * 10}
        for district in districts for date in ("2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01")
    ])
    inputs = {}
    for market in ("rental", "sales"):
        inputs[market] = tmp_path / f"{market}_imputed.csv"
        panel.to_csv(inputs[market], index=False)

    developer = pd.DataFrame([
        {"Project Name": f"PROJECT {district}", "district": district,
         "Reference Month": date, "Total Number of Units in Project": 100,
         "Cumulative Units Sold to-date": sold,
         "Units Launched in the Month": 5,
         "Units Sold in the Month": sold,
         "Cumulative Units Launched to-date": 100,
         "Cumulative Units Launched but Unsold": 100 - sold,
         "Total Number of Unsold Units": 100 - sold,
         "Median Price ($ PSF) in the Month": 1000,
         "Lowest Price ($ PSF) in the Month": 900,
         "Highest Price ($ PSF) in the Month": 1100}
        for district in districts for date, sold in (("Jan 2020", 20), ("Apr 2020", 100))
    ])
    numeric_details = [
        "Total Units in Project by Construction Status Under Construction",
        "Total Units in Project by Completion and Sale Status Uncompleted Units With Pre-Requisites Not Launched Yet",
        "Total Units in Project by Completion and Sale Status Uncompleted Units With Pre-Requisites Launched but Unsold",
        "Total Units in Project by Completion and Sale Status Completed Units With Pre-Requisites Completed but Unsold",
    ]
    details = pd.DataFrame([
        {"Project Name": f"PROJECT {district}", "district": district,
         "year": 2020, "quarter": quarter, **dict.fromkeys(numeric_details, 10)}
        for district in districts for quarter in (1, 2)
    ])
    monkeypatch.setattr(supply, "load_supply_table",
                        lambda path, kind: (developer if kind == "developer_sales" else details).copy())
    monkeypatch.setattr(supply, "display", lambda *args: None)

    def output_folder(name):
        folder = tmp_path / name
        folder.mkdir(exist_ok=True)
        return folder

    settings = SimpleNamespace(prep_path=lambda name: tmp_path / name,
                               folder=output_folder, read_imputed=inputs.__getitem__)
    supply.run(settings)
    for market in ("rental", "sales"):
        result = pd.read_csv(tmp_path / "03_features" / f"{market}_supply.csv")
        pd.testing.assert_frame_equal(result[panel.columns], panel)
        january = result["date"].eq("2020-01-01")
        assert result.loc[january, "project_sold_share_lag1m"].isna().all()
        assert result.loc[~january, "project_sold_share_lag1m"].eq(0.2).all()
