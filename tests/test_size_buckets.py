"""Fixed area boundaries must be identical across markets and input orders."""

import numpy as np
import pandas as pd
import pytest

from functions.imputation.core import export_bucket_ranges, make_size_bucket


def test_exact_boundaries_and_values_immediately_above_them():
    areas = pd.Series([350, 650, np.nextafter(650.0, np.inf), 950,
                       np.nextafter(950.0, np.inf), 1150,
                       np.nextafter(1150.0, np.inf), 1450,
                       np.nextafter(1450.0, np.inf), 10000])
    assert make_size_bucket(areas).tolist() == [
        "SZ1", "SZ1", "SZ2", "SZ2", "SZ3", "SZ3", "SZ4", "SZ4", "SZ5", "SZ5",
    ]


def test_reordering_tied_areas_does_not_change_transaction_assignments():
    areas = pd.Series([500, 650, 650, 650, 950, 950, 1150, 1450, 1450, 1800],
                      index=list("abcdefghij"))
    expected = make_size_bucket(areas)
    actual = make_size_bucket(areas.sample(frac=1, random_state=42)).reindex(areas.index)
    pd.testing.assert_series_equal(actual, expected)
    assert expected.loc[["b", "c", "d"]].eq("SZ1").all()


def test_assignments_do_not_depend_on_sample_distribution_or_available_sizes():
    assert make_size_bucket(pd.Series([1000])).tolist() == ["SZ3"]
    assert make_size_bucket(pd.Series([1000] * 100)).eq("SZ3").all()
    assert make_size_bucket(pd.Series([100, 1000, 100000])).tolist() == ["SZ1", "SZ3", "SZ5"]


def test_exported_rules_reproduce_assignments_with_empty_buckets_and_new_extremes(tmp_path):
    sample = pd.Series([500, 650, 650, 2000])
    ranges = export_bucket_ranges(sample, make_size_bucket(sample))
    assert ranges["size_bucket"].tolist() == ["SZ1", "SZ2", "SZ3", "SZ4", "SZ5"]
    path = tmp_path / "ranges.csv"
    ranges.to_csv(path, index=False)
    restored = pd.read_csv(path)
    assert restored["upper_bound"].iloc[:-1].tolist() == [650, 950, 1150, 1450]
    areas = pd.Series([1, 650, 651, 950, 951, 1150, 1151, 1450, 1451, 100000, np.nan])
    bins = [restored["lower_bound"].iloc[0]] + restored["upper_bound"].tolist()
    replayed = pd.cut(areas, bins=bins, labels=restored["size_bucket"],
                      right=True, include_lowest=True).astype(object).fillna("SZ1")
    pd.testing.assert_series_equal(replayed, make_size_bucket(areas))


def test_legacy_missing_area_fallback_and_index_are_preserved():
    areas = pd.Series([np.nan, 650, 1451], index=[4, 4, 7])
    result = make_size_bucket(areas)
    assert result.index.equals(areas.index)
    assert result.tolist() == ["SZ1", "SZ1", "SZ5"]


def test_conflicting_reference_cut_points_are_rejected():
    from functions.imputation.core import validate_bucket_ranges

    legacy = pd.DataFrame({"size_bucket": [f"SZ{i}" for i in range(1, 6)],
                           "lower_bound": [350, 650, 950, 1150, 1450],
                           "upper_bound": [650, 950, 1150, 1450, 3250]})
    validate_bucket_ranges(legacy)
    legacy.loc[0, "upper_bound"] = 700
    with pytest.raises(ValueError, match="fixed size-bucket"):
        validate_bucket_ranges(legacy)


def test_fixed_five_bucket_policy_rejects_a_different_bucket_count():
    with pytest.raises(ValueError, match="five"):
        make_size_bucket(pd.Series([500, 1000]), q=2)


@pytest.mark.parametrize("market", ["rental", "sales"])
def test_market_orchestration_uses_shared_boundaries_in_monthly_targets(tmp_path, monkeypatch, market):
    import importlib
    import torch
    from functions.config import Settings
    from functions.imputation.artifacts import artifact_paths

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(1)
    settings = Settings(source_root=tmp_path / "data", output_root=tmp_path / "out",
                        imputation_max_configs=1, imputation_max_epochs=1, mc_draws=2)
    settings.folder("01_preprocessing")
    settings.folder("02_imputation")
    export_bucket_ranges().to_csv(settings.bucket_ranges(), index=False)
    # Tie two observations at 650; cover every exact cut and the open upper bin.
    cases = [(650, "SZ1", 0), (650, "SZ1", 1), (950, "SZ2", 2),
             (1150, "SZ3", 3), (1450, "SZ4", 4), (1451, "SZ5", 5)]
    records, expected = [], []
    for district in sorted(set(range(1, 29)) - {24}):
        for date in pd.date_range("2018-01-01", periods=24, freq="MS"):
            for area, bucket, offset in cases:
                if district == 27 and date.month == 3 and date.year == 2018 and bucket == "SZ5":
                    continue
                psf = (3 + offset / 10 + district / 100 + date.month / 1000
                       if market == "rental" else 1000 + offset * 10 + district + date.month)
                row = {"Project Name": f"CONDO {district}", "Postal District": district,
                       "Property Type": "Non-landed Properties" if market == "rental" else "Condominium",
                       "Condo_Age_2026": 15}
                if market == "rental":
                    row.update({"Floor Area (SQFT)": f"{area} to {area}",
                                "Monthly Rent ($)": psf * area,
                                "Lease Commencement Date": date.strftime("%Y-%m-%d")})
                else:
                    row.update({"Area (SQFT)": area, "Unit Price ($ PSF)": psf,
                                "Transacted Price ($)": psf * area,
                                "Sale Date": date.strftime("%Y-%m-%d")})
                records.append(row)
                expected.append({"condo_name": f"CONDO {district}", "size_tier": bucket,
                                 "date": date, "target": psf})
    pd.DataFrame(records).sample(frac=1, random_state=42).to_csv(settings.prepared(market), index=False)
    importlib.import_module(f"functions.imputation.{market}").run(settings)
    output = pd.read_csv(settings.imputed(market), parse_dates=["date"])
    prefix = "rent" if market == "rental" else "sale"
    expected = pd.DataFrame(expected).groupby(["condo_name", "size_tier", "date"])["target"].mean().sort_index()
    observed = output.loc[output["was_observed"]].set_index(["condo_name", "size_tier", "date"])
    assert len(output) == 27 * 5 * 24
    pd.testing.assert_index_equal(observed.sort_index().index, expected.index)
    np.testing.assert_allclose(observed[f"{prefix}_psf_obs"].sort_index(), expected, rtol=1e-6)
    np.testing.assert_array_equal(observed[f"{prefix}_psf_imp"], observed[f"{prefix}_psf_obs"])
    assert output[f"{prefix}_psf_imp"].notna().all()
    for key in ("bundle", "arrays", "metrics", "weights"):
        assert artifact_paths(settings, market)[key].is_file()
    pd.testing.assert_frame_equal(pd.read_csv(settings.bucket_ranges()), export_bucket_ranges())
