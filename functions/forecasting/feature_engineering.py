"""
feature_engineering.py
-----------------------
Lag-feature construction, categorical encoding, and feature-list assembly.

Designed to be target-agnostic: the same functions work for rental PSF,
sales PSF, or any other numeric panel column.  The rental-prediction defaults
are preserved so existing notebooks require no changes.

Public API
----------
lag_names(suffix)                       : generate the 10 lag column names for any suffix
build_lag_features(df, source_col, suffixes)
    Build glob/district/condo lag families from any panel column.
    Default: source_col='rent_psf_imp', suffixes=('', '_t12', '_t24')
build_obs_lag_features(df, source_col)
    Thin wrapper — same as build_lag_features with obs-style suffixes.
    Default: source_col='rent_psf_obs'
encode_categoricals(df, extra_cat_cols) : cast to category dtype
make_feature_lists(df, cat_features, flags, structural_features)
    Return dict of all FEATURES_* lists.
    Pass structural_features to override the default rental feature set.

Constants
---------
BASE_STRUCTURAL  : property / location / macro features shared across targets
LAG1_IMP  … LAG24_OBS : pre-built lag-name lists for the rental use-case
    (generated via lag_names(); re-create with lag_names(your_suffix) for others)

Notes
-----
After FIX #1 all _t12 / _t24 lag columns are numerically identical to the
base columns (all use shift(1) from the same source series).  They exist under
separate names so training cells can reference them consistently by horizon.
"""

import numpy as np
import pandas as pd

from .data_loader import (
    MACRO_COLS_EXTENDED,
    AVG_NEW_SALE_PSF_LAG1Q_COL,
    AVG_RESALE_PSF_LAG1Q_COL,
)


# ── Lag-name generator ────────────────────────────────────────────────────────


def lag_names(suffix: str = "") -> list[str]:
    """
    Return the 10 standard lag column names for a given suffix.

    Examples
    --------
    lag_names('')        → ['glob_med_1m', ..., 'condo_level_12m']
    lag_names('_obs')    → ['glob_med_1m_obs', ..., 'condo_level_12m_obs']
    lag_names('_t12')    → ['glob_med_1m_t12', ..., 'condo_level_12m_t12']
    lag_names('_sales')  → ['glob_med_1m_sales', ..., 'condo_level_12m_sales']
    """
    return [
        f"glob_med_1m{suffix}",
        f"glob_med_3m{suffix}",
        f"glob_med_12m{suffix}",
        f"glob_trend{suffix}",
        f"dist_med_1m{suffix}",
        f"dist_med_3m{suffix}",
        f"dist_med_12m{suffix}",
        f"dist_trend{suffix}",
        f"condo_level_1m{suffix}",
        f"condo_level_12m{suffix}",
    ]


# Pre-built lists for the rental use-case (backward compatibility)
LAG1_IMP = lag_names("")
LAG1_OBS = lag_names("_obs")
LAG12_IMP = lag_names("_t12")
LAG12_OBS = lag_names("_obs_t12")
LAG24_IMP = lag_names("_t24")
LAG24_OBS = lag_names("_obs_t24")


# Planning-Area monthly New Sale / Resale market features. These are built
# from raw monthly averages and shared by all horizons at the same origin.
NEW_RESALE_PRICE_FEATURES = [
    "avg_new_sale_psf_trailing3m",
    "avg_resale_psf_trailing3m",
    "avg_new_sale_psf_trailing12m",
    "avg_resale_psf_trailing12m",
    "new_resale_log_premium_3m",
    "new_resale_log_premium_12m",
]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _month_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Return a calendar month grid covering df's date range."""
    grid = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max(), freq="MS")})
    grid["period"] = grid["date"].dt.year * 100 + grid["date"].dt.month
    return grid


def _build_glob_lags(raw_series: pd.Series, grid: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    g = (
        grid.merge(
            raw_series.reset_index().rename(columns={raw_series.name: "_raw"}),
            on="period",
            how="left",
        )
        .sort_values("period")
        .reset_index(drop=True)
    )
    g[f"glob_med_1m{suffix}"] = g["_raw"].shift(1)
    g[f"glob_med_3m{suffix}"] = g["_raw"].shift(1).rolling(3, min_periods=1).mean()
    g[f"glob_med_12m{suffix}"] = g["_raw"].shift(1).rolling(12, min_periods=3).mean()
    g[f"glob_trend{suffix}"] = g[f"glob_med_1m{suffix}"] - g[f"glob_med_12m{suffix}"]
    glob_lag_cols = [c for c in lag_names(suffix) if c.startswith("glob_")]
    return g[["period"] + glob_lag_cols]


def _build_dist_lags(
    raw_by_district: pd.DataFrame, grid: pd.DataFrame, districts, suffix: str = ""
) -> pd.DataFrame:
    parts = []
    for d in districts:
        g = grid[["period"]].copy()
        g["district"] = d
        g = g.merge(
            raw_by_district[raw_by_district["district"] == d][["period", "_raw"]],
            on="period",
            how="left",
        ).sort_values("period")
        g[f"dist_med_1m{suffix}"] = g["_raw"].shift(1)
        g[f"dist_med_3m{suffix}"] = g["_raw"].shift(1).rolling(3, min_periods=1).mean()
        g[f"dist_med_12m{suffix}"] = g["_raw"].shift(1).rolling(12, min_periods=3).mean()
        g[f"dist_trend{suffix}"] = g[f"dist_med_1m{suffix}"] - g[f"dist_med_12m{suffix}"]
        parts.append(g)
    dist_lag_cols = [c for c in lag_names(suffix) if c.startswith("dist_")]
    return pd.concat(parts, ignore_index=True)[["district", "period"] + dist_lag_cols]


def _build_condo_lags(
    raw_by_condo: pd.DataFrame, grid: pd.DataFrame, projects, suffix: str = ""
) -> pd.DataFrame:
    parts = []
    for pid in projects:
        g = grid[["period"]].copy()
        g["project_idx"] = pid
        g = g.merge(
            raw_by_condo[raw_by_condo["project_idx"] == pid][["period", "_raw"]],
            on="period",
            how="left",
        ).sort_values("period")
        g[f"condo_level_1m{suffix}"] = g["_raw"].shift(1)
        g[f"condo_level_12m{suffix}"] = g["_raw"].shift(1).rolling(12, min_periods=3).mean()
        parts.append(g)
    condo_lag_cols = [c for c in lag_names(suffix) if c.startswith("condo_")]
    return pd.concat(parts, ignore_index=True)[["project_idx", "period"] + condo_lag_cols]


# ── Public: generic lag builder ───────────────────────────────────────────────


def add_post_covid_dummy(
    df: pd.DataFrame,
    cutoff: int = 202201,
    col: str = "post_covid",
) -> pd.DataFrame:
    """
    Add a binary post-COVID dummy based on the **observation period**.

    ``post_covid = 1`` when ``period >= cutoff`` (default 202201), else 0.
    Must be called **before** ``build_lag_features`` so the column exists in the
    panel before any lag-shifted features are computed.  The column reflects the
    origin month of each row, not any forward target period.

    Parameters
    ----------
    cutoff : yyyymm threshold (inclusive).  Default 202201 (Jan 2022).
    col    : output column name.  Default 'post_covid'.
    """
    df = df.copy()
    df[col] = (df["period"].astype(int) >= cutoff).astype(np.int8)
    n_post = int(df[col].sum())
    print(f"  post_covid dummy added: {n_post:,} rows with period >= {cutoff} (out of {len(df):,})")
    return df


def build_lag_features(
    df: pd.DataFrame,
    source_col: str = "rent_psf_imp",
    suffixes: tuple = ("", "_t12", "_t24"),
) -> pd.DataFrame:
    """
    Build glob / district / condo-level lag features from any numeric panel column.

    Each element of *suffixes* produces one complete family of 10 lag columns.
    After FIX #1 these families are numerically identical; they exist under
    separate names so training cells can reference them by horizon.

    Parameters
    ----------
    source_col : column to aggregate across projects/districts/globally.
                 Default 'rent_psf_imp'.  Change to 'rent_psf_obs',
                 'sales_psf', 'sales_price', etc. as needed.
    suffixes   : column-name suffix for each horizon family.
                 Default ('', '_t12', '_t24') → glob_med_1m, glob_med_1m_t12, …
                 For an observed-only stack: ('_obs', '_obs_t12', '_obs_t24')
                 For a custom target:        ('_sales', '_sales_t12', '_sales_t24')

    Returns
    -------
    df with new lag columns merged in (new DataFrame, original not mutated).
    """
    if source_col not in df.columns:
        raise ValueError(f"source_col '{source_col}' not found in df.columns")

    grid = _month_grid(df)
    projects = df["project_idx"].unique()
    districts = df["district"].dropna().unique()

    glob_raw = df.groupby("period")[source_col].median().rename("_raw")
    dist_raw = (
        df.groupby(["district", "period"])[source_col]
        .median()
        .reset_index()
        .rename(columns={source_col: "_raw"})
    )
    condo_raw = (
        df.groupby(["project_idx", "period"])[source_col]
        .median()
        .reset_index()
        .rename(columns={source_col: "_raw"})
    )

    for sfx in suffixes:
        glob_lags = _build_glob_lags(glob_raw, grid, suffix=sfx)
        dist_lags = _build_dist_lags(dist_raw, grid, districts, suffix=sfx)
        condo_lags = _build_condo_lags(condo_raw, grid, projects, suffix=sfx)

        df = df.merge(glob_lags, on="period", how="left")
        df = df.merge(dist_lags, on=["district", "period"], how="left")
        df = df.merge(condo_lags, on=["project_idx", "period"], how="left")

        # Condo fallback: district median → global median
        df[f"condo_level_1m{sfx}"] = (
            df[f"condo_level_1m{sfx}"]
            .fillna(df[f"dist_med_1m{sfx}"])
            .fillna(df[f"glob_med_1m{sfx}"])
        )
        df[f"condo_level_12m{sfx}"] = (
            df[f"condo_level_12m{sfx}"]
            .fillna(df[f"dist_med_12m{sfx}"])
            .fillna(df[f"glob_med_12m{sfx}"])
        )

    suffix_str = " / ".join(f"'{s}'" for s in suffixes)
    print(f"Lag features built from '{source_col}'  suffixes: {suffix_str}")
    return df


def build_obs_lag_features(
    df: pd.DataFrame,
    source_col: str = "rent_psf_obs",
) -> pd.DataFrame:
    """
    Convenience wrapper: build observed-source lag families (_obs / _obs_t12 / _obs_t24).
    Change source_col for other observed-only columns (e.g. 'sales_psf_obs').
    """
    return build_lag_features(
        df,
        source_col=source_col,
        suffixes=("_obs", "_obs_t12", "_obs_t24"),
    )


def build_new_resale_price_features(
    df: pd.DataFrame,
    area_col: str = "Planning Area",
    period_col: str = "period",
    new_sale_col: str = "avg_new_sale_psf",
    resale_col: str = "avg_resale_psf",
    data_lag_months: int = 1,
    min_periods_3m: int = 2,
    min_periods_12m: int = 6,
) -> pd.DataFrame:
    """Build Planning-Area trailing PSF levels and log premiums.

    The panel repeats each Planning-Area/month value across project/size rows.
    This helper first collapses to one row per area/month, reindexes each area
    to a complete calendar grid, creates the rolling features, and only then
    merges them back. ``data_lag_months=1`` makes the origin-t windows end at
    t-1; use 0 only if origin-month transactions are available at prediction.
    """
    required = {area_col, period_col, new_sale_col, resale_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Cannot build New Sale / Resale price features; missing columns: {missing}")
    if not isinstance(data_lag_months, int) or data_lag_months < 0:
        raise ValueError("data_lag_months must be a non-negative integer")
    if not 1 <= min_periods_3m <= 3:
        raise ValueError("min_periods_3m must be between 1 and 3")
    if not 1 <= min_periods_12m <= 12:
        raise ValueError("min_periods_12m must be between 1 and 12")

    raw_cols = [new_sale_col, resale_col]
    consistency = df.groupby([area_col, period_col], observed=True)[raw_cols].nunique(dropna=True)
    if (consistency > 1).any().any():
        bad = consistency[(consistency > 1).any(axis=1)].head()
        raise ValueError(
            "Multiple raw PSF values found for the same Planning Area x "
            f"period. Example keys:\n{bad}"
        )

    monthly = df.groupby([area_col, period_col], observed=True, as_index=False)[raw_cols].first()
    monthly["_month_start"] = pd.to_datetime(
        monthly[period_col].astype("Int64").astype(str),
        format="%Y%m",
        errors="coerce",
    )
    if monthly["_month_start"].isna().any():
        bad_periods = (
            monthly.loc[monthly["_month_start"].isna(), period_col].drop_duplicates().tolist()[:10]
        )
        raise ValueError(f"Invalid YYYYMM periods: {bad_periods}")

    area_values = monthly[area_col].drop_duplicates().tolist()
    month_values = pd.date_range(
        monthly["_month_start"].min(),
        monthly["_month_start"].max(),
        freq="MS",
    )
    calendar = pd.MultiIndex.from_product(
        [area_values, month_values],
        names=[area_col, "_month_start"],
    )
    monthly = (
        monthly.drop(columns=period_col)
        .set_index([area_col, "_month_start"])
        .reindex(calendar)
        .sort_index()
    )

    for source_col, output_prefix in (
        (new_sale_col, "avg_new_sale_psf"),
        (resale_col, "avg_resale_psf"),
    ):
        shifted = monthly.groupby(level=0, observed=True)[source_col].shift(data_lag_months)
        monthly[f"{output_prefix}_trailing3m"] = (
            shifted.groupby(level=0, observed=True)
            .rolling(3, min_periods=min_periods_3m)
            .mean()
            .reset_index(level=0, drop=True)
        )
        monthly[f"{output_prefix}_trailing12m"] = (
            shifted.groupby(level=0, observed=True)
            .rolling(12, min_periods=min_periods_12m)
            .mean()
            .reset_index(level=0, drop=True)
        )

    valid_3m = monthly["avg_new_sale_psf_trailing3m"].gt(0) & monthly[
        "avg_resale_psf_trailing3m"
    ].gt(0)
    monthly["new_resale_log_premium_3m"] = np.nan
    monthly.loc[valid_3m, "new_resale_log_premium_3m"] = np.log(
        monthly.loc[valid_3m, "avg_new_sale_psf_trailing3m"]
    ) - np.log(monthly.loc[valid_3m, "avg_resale_psf_trailing3m"])

    valid_12m = monthly["avg_new_sale_psf_trailing12m"].gt(0) & monthly[
        "avg_resale_psf_trailing12m"
    ].gt(0)
    monthly["new_resale_log_premium_12m"] = np.nan
    monthly.loc[valid_12m, "new_resale_log_premium_12m"] = np.log(
        monthly.loc[valid_12m, "avg_new_sale_psf_trailing12m"]
    ) - np.log(monthly.loc[valid_12m, "avg_resale_psf_trailing12m"])

    derived = monthly[NEW_RESALE_PRICE_FEATURES].reset_index()
    derived[period_col] = (
        derived["_month_start"].dt.year * 100 + derived["_month_start"].dt.month
    ).astype(int)
    derived = derived.drop(columns="_month_start")
    if derived.duplicated([area_col, period_col]).any():
        raise AssertionError("Derived price-feature keys are not unique")

    out = df.drop(
        columns=[c for c in NEW_RESALE_PRICE_FEATURES if c in df.columns],
        errors="ignore",
    ).copy()
    out["_price_feature_row_order"] = np.arange(len(out))
    out = out.merge(
        derived,
        on=[area_col, period_col],
        how="left",
        validate="many_to_one",
        sort=False,
    ).sort_values("_price_feature_row_order")
    if len(out) != len(df):
        raise AssertionError("Row count changed while merging price features")
    if not np.array_equal(out["_price_feature_row_order"].to_numpy(), np.arange(len(df))):
        raise AssertionError("Row order changed while merging price features")
    out = out.drop(columns="_price_feature_row_order")

    coverage = {col: f"{out[col].notna().mean():.1%}" for col in NEW_RESALE_PRICE_FEATURES}
    print(f"New Sale / Resale price features built (data_lag_months={data_lag_months}): {coverage}")
    return out


# ── Public: categorical encoding ─────────────────────────────────────────────


def encode_categoricals(
    df: pd.DataFrame,
    extra_cat_cols: list | None = None,
) -> tuple:
    """
    Cast category columns to pandas category dtype.

    Parameters
    ----------
    extra_cat_cols : additional column names to treat as categorical,
                     e.g. ['property_type'] for a sales-price dataset.

    Returns (df, CAT_FEATURES) where CAT_FEATURES lists columns present in df.
    """
    df["Postal District"] = df["district"].astype(str)
    base_cat = ["Postal District", "Planning Area", "Neighbourhood", "size_tier", "month_of_year"]
    all_cat = base_cat + (extra_cat_cols or [])
    cat_features = [c for c in all_cat if c in df.columns]
    for col in cat_features:
        df[col] = df[col].astype("category")
    print(f"Categorical features: {cat_features}")
    return df, cat_features


# ── Public: feature list assembly ─────────────────────────────────────────────

# Non-lag structural features — property + location + macros.
# Override via structural_features parameter in make_feature_lists() when the
# target dataset has different columns (e.g. no 'area_sqft_med' in transaction data).
BASE_STRUCTURAL = [
    "area_sqft_med",
    "condo_age_at_transaction",
    "post_covid",
    "tenure_freehold_like",
    "tenure_medium_lease_60_80",
    "tenure_more_than_80",
    "tenure_short_lease_lt60",
    "tenure_unknown",
    "tenure_remaining_years",
    "project_size_small",
    "project_size_medium",
    "project_size_large",
    "Large_Dev_200plus",
    "Postal District",
    "Planning Area",
    "nearest_mrt_dist",
    "nearest_bus_stops_dist",
    "n_bus_stops_within_500m",
    "nearest_supermarkets_dist",
    "n_supermarkets_within_500m",
    "nearest_parks_dist",
    "n_parks_within_500m",
    "nearest_clinics_dist",
    "n_clinics_within_500m",
    "nearest_bank_dist",
    "n_bank_within_500m",
    "nearest_atms_dist",
    "n_atms_within_500m",
    "nearest_school_dist",
    "month_of_year",
    # ── Consulting expert forecast ──────────────────────────────────────────
    "consulting_rental_forecast_mid",
    "consulting_rental_direction",
    "consulting_sale_forecast_mid",
    "consulting_sale_direction",
    # Core monthly macros (9) — merged by date then extra calendar-month lag in fill_missing
    # (MACRO_COLS_CORE + CORE_MACRO_EXTRA_PANEL_SHIFT_MONTHS via lease period; see data_loader).
    "ura_private_rental_index_yoy_pct",
    "sora_compounded_3m_pct",
    "gva_yoy_growth_pct",
    "cpi_all_items_infl_yoy_pct",
    "unemployment_rate_sa_pct",
    "sg_govt_bond_yield_10y_pct",
    "sgd_per_usd_logret_12m_pct",
    "ura_private_price_index_avg_3seg_yoy_pct",
    "hdb_resale_price_index_yoy_pct",
    # New Sale / Resale avg PSF levels — already lag1q at source, per Planning Area.
    # Replaces the earlier ratio-based DISCOUNT_LAG1Q_COL for this experiment
    # branch: kept as two independent raw levels instead of a single ratio
    # (see avg_psf_feature_quarter_to_sales.ipynb notes for rationale).
    AVG_NEW_SALE_PSF_LAG1Q_COL,
    AVG_RESALE_PSF_LAG1Q_COL,
    *NEW_RESALE_PRICE_FEATURES,
    # Extended macro/supply/labour candidates, with their respective source lags.
] + MACRO_COLS_EXTENDED


def make_feature_lists(
    df: pd.DataFrame,
    cat_features: list,
    flags: dict,
    structural_features: list | None = None,
    lag1_primary: list | None = None,
    lag1_secondary: list | None = None,
    lag12_primary: list | None = None,
    lag12_secondary: list | None = None,
    lag24_primary: list | None = None,
    lag24_secondary: list | None = None,
) -> dict:
    """
    Build all FEATURES_* lists used by the training cells.

    Parameters
    ----------
    df, cat_features : panel and categorical column names
    flags            : dict with keys
        use_obs_h1  (bool) : h=1  imputed model uses secondary lags? default False
        use_obs_h12 (bool) : h=12 imputed model uses secondary lags? default True
        use_obs_h24 (bool) : h=24 imputed model uses secondary lags? default True
        h1_noimp_core_macros_only (bool) : drop extended macros from h=1 secondary?
                                            default False
        h12_noimp_core_macros_only (bool) : drop extended macros from h=12 secondary?
                                            default False
        h24_noimp_core_macros_only (bool) : drop extended macros from h=24 secondary?
                                            default True

    structural_features : non-lag features to include.  Defaults to BASE_STRUCTURAL.
        Override for datasets with different columns, e.g. omit 'area_sqft_med' if
        the panel is at transaction level, or add 'property_type' for sales data.

    lag*_primary / lag*_secondary : lag column-name lists for the primary (imputed)
        and secondary (observed-only) model variants at each horizon.
        Defaults to LAG1_IMP/LAG1_OBS, LAG12_IMP/LAG12_OBS, LAG24_IMP/LAG24_OBS.
        For a custom target column, use lag_names(your_suffix) to generate these.

    Candidates absent from the panel are reported once before returning. These
    are availability-based lists; market/horizon-specific exclusions still apply.

    Returns
    -------
    dict with keys:
        FEATURES_A, CAT_FEATURES_A,
        FEATURES_OBS, CAT_FEATURES_OBS,
        FEATURES_T12, FEATURES_T12_OBS,
        FEATURES_T24, FEATURES_T24_OBS,
        LAG1_COLS, LAG12_COLS, LAG24_COLS, HAS_COLS
    """
    # ── Resolve defaults ─────────────────────────────────────────────────────
    use_obs_h1 = flags.get("use_obs_h1", False)
    use_obs_h12 = flags.get("use_obs_h12", True)
    use_obs_h24 = flags.get("use_obs_h24", True)
    h1_core = flags.get("h1_noimp_core_macros_only", False)
    h12_core = flags.get("h12_noimp_core_macros_only", False)
    h24_core = flags.get("h24_noimp_core_macros_only", True)

    struct = structural_features if structural_features is not None else BASE_STRUCTURAL
    L1p = lag1_primary if lag1_primary is not None else LAG1_IMP
    L1s = lag1_secondary if lag1_secondary is not None else LAG1_OBS
    L12p = lag12_primary if lag12_primary is not None else LAG12_IMP
    L12s = lag12_secondary if lag12_secondary is not None else LAG12_OBS
    L24p = lag24_primary if lag24_primary is not None else LAG24_IMP
    L24s = lag24_secondary if lag24_secondary is not None else LAG24_OBS

    has_cols = sorted([c for c in df.columns if c.startswith("Has_")])

    # ── Base h=1 primary (imputed) feature list ───────────────────────────────
    FEATURES_A = list(dict.fromkeys(struct + L1p + has_cols))
    FEATURES_A = [c for c in FEATURES_A if c in df.columns]
    if use_obs_h1:
        _swap = dict(zip(L1p, L1s))
        FEATURES_A = [_swap.get(f, f) for f in FEATURES_A]
        FEATURES_A = [c for c in FEATURES_A if c in df.columns]
    CAT_A = [c for c in cat_features if c in FEATURES_A]

    # ── Secondary (observed-only) h=1 baseline ────────────────────────────────
    FEATURES_OBS = [f for f in FEATURES_A if f not in L1p] + L1s
    FEATURES_OBS = [f for f in FEATURES_OBS if f in df.columns]
    CAT_OBS = [c for c in cat_features if c in FEATURES_OBS]

    lag1_actual = L1s if use_obs_h1 else L1p
    L12_use = L12s if use_obs_h12 else L12p
    L24_use = L24s if use_obs_h24 else L24p

    # ── h=12 ─────────────────────────────────────────────────────────────────
    FEATURES_T12 = list(dict.fromkeys([f for f in FEATURES_A if f not in lag1_actual] + L12_use))
    FEATURES_T12 = [f for f in FEATURES_T12 if f in df.columns]

    FEATURES_T12_OBS = list(dict.fromkeys([f for f in FEATURES_OBS if f not in L1s] + L12s))
    FEATURES_T12_OBS = [f for f in FEATURES_T12_OBS if f in df.columns]

    if h12_core:
        n_before = len(FEATURES_T12_OBS)
        FEATURES_T12_OBS = [f for f in FEATURES_T12_OBS if f not in MACRO_COLS_EXTENDED]
        print(
            f"h=12 secondary: core macros only; dropped "
            f"{n_before - len(FEATURES_T12_OBS)} extended cols "
            f"({n_before} → {len(FEATURES_T12_OBS)} features)"
        )

    # ── h=24 ─────────────────────────────────────────────────────────────────
    FEATURES_T24 = list(dict.fromkeys([f for f in FEATURES_A if f not in lag1_actual] + L24_use))
    FEATURES_T24 = [f for f in FEATURES_T24 if f in df.columns]

    FEATURES_T24_OBS = list(dict.fromkeys([f for f in FEATURES_OBS if f not in L1s] + L24s))
    FEATURES_T24_OBS = [f for f in FEATURES_T24_OBS if f in df.columns]

    if h24_core:
        n_before = len(FEATURES_T24_OBS)
        FEATURES_T24_OBS = [f for f in FEATURES_T24_OBS if f not in MACRO_COLS_EXTENDED]
        print(
            f"h=24 secondary: core macros only; dropped "
            f"{n_before - len(FEATURES_T24_OBS)} extended cols "
            f"({n_before} → {len(FEATURES_T24_OBS)} features)"
        )

    if h1_core:
        n_before = len(FEATURES_OBS)
        FEATURES_OBS = [f for f in FEATURES_OBS if f not in MACRO_COLS_EXTENDED]
        CAT_OBS = [c for c in cat_features if c in FEATURES_OBS]
        print(
            f"h=1 secondary: core macros only; dropped "
            f"{n_before - len(FEATURES_OBS)} extended cols "
            f"({n_before} → {len(FEATURES_OBS)} features)"
        )

    candidates = list(dict.fromkeys(struct + L1p + L1s + L12_use + L12s + L24_use + L24s))
    missing = [name for name in candidates if name not in df.columns]
    if missing:
        print(
            f"[forecast features] WARNING: {len(missing)} unavailable candidates skipped: "
            + ", ".join(missing)
        )

    print(f"FEATURES_A      : {len(FEATURES_A)}  ({len(CAT_A)} categorical)")
    print(f"FEATURES_OBS    : {len(FEATURES_OBS)}")
    print(f"FEATURES_T12    : {len(FEATURES_T12)}")
    print(f"FEATURES_T12_OBS: {len(FEATURES_T12_OBS)}")
    print(f"FEATURES_T24    : {len(FEATURES_T24)}")
    print(f"FEATURES_T24_OBS: {len(FEATURES_T24_OBS)}")

    return dict(
        FEATURES_A=FEATURES_A,
        CAT_FEATURES_A=CAT_A,
        FEATURES_OBS=FEATURES_OBS,
        CAT_FEATURES_OBS=CAT_OBS,
        FEATURES_T12=FEATURES_T12,
        FEATURES_T12_OBS=FEATURES_T12_OBS,
        FEATURES_T24=FEATURES_T24,
        FEATURES_T24_OBS=FEATURES_T24_OBS,
        LAG1_COLS=lag1_actual,
        LAG12_COLS=L12_use,
        LAG24_COLS=L24_use,
        HAS_COLS=has_cols,
    )
