"""Repair missing rental macro cells from exact calendar-month reference keys."""

import pandas as pd


RENTAL_MONTHLY_REFERENCE_LAG_MONTHS = 1
RENTAL_MONTHLY_MACRO_COLS = (
    "cpi_all_items_infl_yoy_pct",
    "unemployment_rate_sa_pct",
    "sora_compounded_3m_pct",
    "sg_govt_bond_yield_10y_pct",
    "sgd_per_usd_logret_12m_pct",
    "ura_private_rental_index_yoy_pct",
    "ura_private_price_index_avg_3seg_yoy_pct",
    "hdb_resale_price_index_yoy_pct",
    "gva_yoy_growth_pct",
)


def fill_rental_monthly_macro_gaps(frame, date_column, source_path, *, copy=True):
    """Fill only unknown cells using reference month t-1; retain known values.

    The one-month offset is the existing rental snapshot convention. It is
    additional to forecasting's unchanged three-month calendar shift. No
    interpolation, forward-fill or future reference observation is used here.
    Runtime callers own their frames and pass copy=False to avoid copying a
    multi-GB panel. Price-only inputs with no monthly macro columns are valid.
    """
    result = frame.copy() if copy else frame
    columns = [name for name in RENTAL_MONTHLY_MACRO_COLS if name in result.columns]
    missing = [name for name in columns if result[name].isna().any()]
    if not missing:
        print("[rental monthly macros] No missing monthly macro cells to repair")
        return result
    if date_column not in result.columns:
        raise KeyError(f"Rental macro repair requires date column {date_column!r}")
    reference = pd.read_csv(source_path)
    required = {"month", *missing}
    absent = required - set(reference.columns)
    if absent:
        raise KeyError(f"Monthly macro reference is missing columns: {sorted(absent)}")
    source_month = pd.to_datetime(reference["month"], errors="raise").dt.to_period("M")
    if source_month.isna().any() or source_month.duplicated().any():
        raise ValueError("Monthly macro reference must have unique, valid month keys")
    lookup = reference[missing].apply(pd.to_numeric, errors="raise")
    lookup.index = source_month
    dates = pd.to_datetime(result[date_column], errors="coerce")
    wanted = dates.dt.to_period("M") - RENTAL_MONTHLY_REFERENCE_LAG_MONTHS
    filled = {}
    for column in missing:
        replacement = wanted.map(lookup[column])
        repair = result[column].isna() & replacement.notna()
        filled[column] = int(repair.sum())
        if filled[column]:
            result[column] = result[column].fillna(replacement)
    print(
        "[rental monthly macros] Exact reference month t-1; filled cells: "
        + str(filled)
    )
    return result
