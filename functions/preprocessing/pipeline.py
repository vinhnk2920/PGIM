"""Prepare both markets using shared macro and outlier functions."""

import gc
import json

import numpy as np
import pandas as pd

from functions.preprocessing.quarterly import add_quarterly_macros
from functions.preprocessing.monthly import fill_rental_monthly_macro_gaps

ASSESSED_FEATURES = [
    "assessed_non_landed_private_residential_owned_by_companies_share_lag1q",
    "assessed_non_landed_private_residential_owned_by_pr_foreigners_share_lag1q",
]


def assessed_reference(settings, dates):
    """Reuse the existing derived series if its original raw download is absent."""
    raw = settings.prep_path(
        "macro",
        "quarterly",
        "Assessed Non Landed Private Residential Units By Locals or Foreigners.csv",
    )
    if raw.exists():
        return None
    from functions.experiment import reported
    if reported(settings):
        raise FileNotFoundError(f"Fresh raw run requires the original ownership source: {raw}")
    reference = (
        settings.source_root / "imputation" / "URA_renting_imputed_vae_a4_mcra_full_districts.csv"
    )
    wanted = set(pd.to_datetime(dates).dt.to_period("M").astype(str))
    print(
        f"[macro reference] Raw assessed-ownership download is absent; reuse derived monthly features from {reference}"
    )
    # The source VAE writes a complete calendar per series, in consecutive blocks.
    chunk = pd.read_csv(
        reference, usecols=["project_idx", "date"] + ASSESSED_FEATURES, nrows=10_000
    )
    first = chunk["project_idx"].iloc[0]
    if chunk["project_idx"].nunique() < 2:
        raise ValueError("Could not identify a complete monthly series in the assessed reference")
    chunk["date"] = pd.to_datetime(chunk["date"]).dt.to_period("M").astype(str)
    first_months = chunk.loc[chunk["project_idx"].eq(first), "date"]
    expected = set(pd.period_range(first_months.min(), first_months.max(), freq="M").astype(str))
    if set(first_months) != expected:
        raise ValueError("Assessed reference has gaps in its monthly grid")
    monthly = chunk.drop(columns="project_idx")
    if monthly.groupby("date")[ASSESSED_FEATURES].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Existing assessed-ownership reference is not constant within a month")
    monthly = monthly.drop_duplicates("date")
    if not wanted <= expected:
        print(
            f"[macro reference] {len(wanted - expected)} transaction months are outside the reference; ownership features remain missing there"
        )
    dt = pd.to_datetime(monthly.pop("date"))
    monthly["Year"], monthly["Month"], monthly["Quarter"] = dt.dt.year, dt.dt.month, dt.dt.quarter
    cache = settings.folder("01_preprocessing") / "assessed_ownership_monthly_reference.csv"
    monthly.to_csv(cache, index=False)
    return monthly


def numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def area_midpoint(value):
    try:
        parts = str(value).replace(",", "").split(" to ")
        return sum(float(p) for p in parts) / len(parts)
    except (TypeError, ValueError):
        return np.nan


def remove_outliers(frame, market):
    """Preserve original property filters and global 1st/99th percentile rule."""
    if market == "rental":
        df = frame.loc[frame["Property Type"].eq("Non-landed Properties")].copy()
        area, amount, psf = "Floor Area (SQFT) mid", "Monthly Rent ($)", "Rent per SQFT"
        df[area] = df["Floor Area (SQFT)"].map(area_midpoint)
        df[amount] = numeric(df[amount])
        df[psf] = df[amount] / df[area]
    else:
        df = frame.loc[
            frame["Property Type"].isin(["Condominium", "Apartment", "Executive Condominium"])
        ].copy()
        area, amount, psf = "Area (SQFT) mid", "Transacted Price ($)", "Unit Price ($ PSF)"
        df[area] = df["Area (SQFT)"].map(area_midpoint)
        df[amount], df[psf] = numeric(df[amount]), numeric(df[psf])
    flag = pd.Series(False, index=df.index)
    thresholds = {}
    for col in (area, amount, psf):
        lo, hi = df[col].quantile([0.01, 0.99])
        thresholds[col] = {"lower_1pct": float(lo), "upper_99pct": float(hi)}
        flag |= df[col].lt(lo) | df[col].gt(hi)
    result = df.loc[~flag].copy()
    if market == "sales":
        result.drop(columns=area, inplace=True)
    audit = dict(
        input_rows=len(frame),
        eligible_rows=len(df),
        removed_rows=int(flag.sum()),
        output_rows=len(result),
        thresholds=thresholds,
    )
    print(f"[{market}] outliers: {len(df):,} eligible -> {len(result):,} retained")
    return result, audit


def resolve_duplicate_columns(df):
    """Original sales rule: prefer the monthly macro (_y) over earlier (_x)."""
    df = df.drop(columns=["Sale_Date"], errors="ignore")
    for col in list(df.columns):
        if col.endswith("_x") and col[:-2] + "_y" in df.columns:
            df = df.drop(columns=col).rename(columns={col[:-2] + "_y": col[:-2]})
    if df.columns.duplicated().any():
        raise ValueError("Duplicate columns remain after sales macro cleanup")
    return df


def run(settings, markets=("rental", "sales"), rebuild_raw=False):
    output = settings.folder("01_preprocessing")
    if rebuild_raw:
        from functions.preprocessing import raw_rental, raw_sales, sales_amenities

        if "rental" in markets:
            raw_rental.run(settings)
        if "sales" in markets:
            missing = settings.prep_path("property", "property_details_facilities.csv")
            if not missing.exists():
                raise FileNotFoundError(
                    f"Raw rebuild needs {missing}. Existing enriched snapshots can be used without --rebuild-raw."
                )
            raw_sales.run(settings)
            sales_amenities.run(settings)
    for market in markets:
        name = "URA_merged_full_v3_new.csv" if market == "rental" else "URA_sale_data.csv"
        source = output / name if rebuild_raw else settings.transaction_input(market, "enriched")
        print(f"[preprocessing 1/3] {market}: load enriched transactions from {source}")
        df = pd.read_csv(source, low_memory=False)
        print(f"[preprocessing 2/3] {market}: merge quarterly macro features")
        date = "Lease Commencement Date" if market == "rental" else "Sale Date"
        from functions.experiment import reported
        if market == "rental" and not reported(settings):
            df = fill_rental_monthly_macro_gaps(
                df, date, settings.prep_path("macro", "monthly", "sg_macro_monthly.csv"),
                copy=False,
            )
        fallback = assessed_reference(settings, df[date])
        df = add_quarterly_macros(
            df,
            date,
            settings.prep_path("macro", "quarterly"),
            assessed_monthly=fallback,
        )
        if market == "sales":
            df = resolve_duplicate_columns(df)
        print(f"[preprocessing 3/3] {market}: filter property types and outliers")
        df, audit = remove_outliers(df, market)
        destination = settings.prepared(market)
        df.to_csv(destination, index=False, chunksize=100_000)
        audit.update(source=str(source), output=str(destination), market=market)
        (output / f"{market}_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"Saved {destination}")
        del df
        gc.collect()
