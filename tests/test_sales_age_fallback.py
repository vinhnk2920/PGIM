"""Sales must load its age fallback without exporting the redundant source field."""

import numpy as np
import pandas as pd
import pytest

from functions.config import Settings


@pytest.mark.parametrize("include_fallback", [True, False])
def test_sales_orchestration_preserves_and_fills_canonical_age(tmp_path, monkeypatch, include_fallback):
    import torch
    from functions.imputation import sales

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(1)
    settings = Settings(source_root=tmp_path / "data", output_root=tmp_path / "out",
                        imputation_max_configs=1, imputation_max_epochs=1, mc_draws=2)
    settings.folder("01_preprocessing")
    settings.folder("02_imputation")
    from functions.imputation.core import export_bucket_ranges

    export_bucket_ranges().to_csv(settings.bucket_ranges(), index=False)
    records = []
    for district in sorted(set(range(1, 29)) - {24}):
        for month in pd.date_range("2018-01-01", periods=24, freq="MS"):
            for bucket in range(5):
                if district == 27 and bucket == 0 and month == pd.Timestamp("2018-03-01"):
                    continue
                record = {"Project Name": f"CONDO {district}", "Postal District": district,
                          "Property Type": "Condominium", "Area (SQFT)": str((500, 800, 1000, 1300, 1700)[bucket]),
                          "Unit Price ($ PSF)": 1000 + district * 10 + bucket + month.month,
                          "Transacted Price ($)": 1_000_000, "Sale Date": month.strftime("%Y-%m-%d"),
                          "Condo_Age_2026": np.nan if district in (1, 2, 3) else 15}
                if include_fallback:
                    record["house_age_2026"] = {1: 56, 2: 0, 3: np.nan}.get(district, 99)
                records.append(record)
    source = pd.DataFrame(records)
    source.to_csv(settings.prepared("sales"), index=False)
    sales.run(settings)
    output = pd.read_csv(settings.imputed("sales"))
    assert len(output) == 3240
    assert "house_age_2026" not in output.columns
    present = output.loc[output["district"].eq(4)]
    assert present["Condo_Age_2026"].eq(15).all()
    assert output.loc[output["district"].eq(3), "Condo_Age_2026"].isna().all()
    fallback = output.loc[output["district"].eq(1)]
    zero_age = output.loc[output["district"].eq(2)]
    if include_fallback:
        assert fallback["Condo_Age_2026"].eq(56).all()
        years = pd.to_datetime(fallback["date"]).dt.year
        pd.testing.assert_series_equal(fallback["condo_age_at_transaction"],
                                       (years - 1970).astype(float), check_names=False)
        assert zero_age["Condo_Age_2026"].eq(0).all()
        assert zero_age["condo_age_at_transaction"].eq(0).all()
    else:
        assert fallback["Condo_Age_2026"].isna().all()
        assert fallback["condo_age_at_transaction"].isna().all()
    observed = output["was_observed"]
    assert observed.sum() == len(source)
    assert output["sale_psf_imp"].notna().all()
    np.testing.assert_array_equal(output.loc[observed, "sale_psf_imp"], output.loc[observed, "sale_psf_obs"])
