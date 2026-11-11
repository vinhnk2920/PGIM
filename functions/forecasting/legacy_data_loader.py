"""
Historical loader for the explicit reported-method replication profile only.
Original comments below are retained as source context; they do not establish
point-in-time validity. In particular, supply variables are aggregated by date.

data_loader.py
--------------
Panel loading and missing-value filling for the VAE-imputed rental dataset.

Public API
----------
load_panel(csv_path)           : load CSV, add period/month_of_year/lease_year
fill_missing(df, test_cutoff)  : fill NaNs (static features + macros + core macro lag policy)

Constants
---------
DROP_MODEL_COLS                   : columns dropped before model fitting (kept in saved CSV)
MACRO_COLS                        : all 30 macro column names
MACRO_COLS_CORE                   : nine monthly macro / index indicators (see lag policy below)
MACRO_COLS_EXTENDED               : twenty-one supply / pipeline indicators (lag1q at source)
CORE_MACRO_EXTRA_PANEL_SHIFT_MONTHS
    Extra **calendar-month** lag applied in ``fill_missing`` **only** to ``MACRO_COLS_CORE``:
    each lease row's ``period`` takes macro values from ``add_months(period, -k)`` (see
    ``_lag_macros_by_calendar_month_inplace``). Not row-based ``shift``, so irregular lease density
    per project does not distort timing. Extended macros are **not** shifted here.

Macro lag policy (plain language)
---------------------------------
1. Core monthly indicators are merged onto each row by ``date``. They should be read as
   **economy-wide time series**, not as if they were known the instant each lease was signed.
2. The upstream construction already embeds roughly **one month** of reporting lag relative
   to treating the macro as literally "current" for that calendar month.
3. ``fill_missing`` step 8 applies **additional** lag on **calendar months** (``period``):
   row ``period`` ``P`` receives core macro levels recorded for ``add_months(P, -k)``
   after step 7's merge/ffill (``k`` = ``CORE_MACRO_EXTRA_PANEL_SHIFT_MONTHS``). This is
   **not** ``groupby.shift`` over lease rows. It is still a modelling choice, not a literal
   real-world publication timestamp.
4. ``MACRO_COLS_EXTENDED`` columns are left as merged after forward-fill / medians only;
   they are already **lag-one-quarter** series at source.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from functions.forecasting.utils import add_months


def _lag_macros_by_calendar_month_inplace(df: pd.DataFrame, cols: list[str], k: int) -> None:
    """
    In-place: assign each row's macro values from *k* **calendar months** earlier.
    Uses lease ``period`` (yyyymm), not ``groupby(project_idx).shift``.
    """
    if k <= 0 or not cols:
        return

    period_int = df['period'].astype(int)
    periods_needed = np.unique(period_int.to_numpy())

    wide = df.groupby('period', sort=False)[cols].first()
    wide.index = wide.index.astype(int)
    idx_set = set(wide.index)

    for col in cols:
        src = wide[col]
        lagged_map = {}
        for p in periods_needed:
            pi = int(p)
            pk = add_months(pi, -k)
            lagged_map[pi] = src.loc[pk] if pk in idx_set else np.nan
        df[col] = period_int.map(lagged_map)


# ── Column lists ──────────────────────────────────────────────────────────────

DROP_MODEL_COLS = [
    'latitude', 'longitude', 'size_tier', 'Neighbourhood',
    'obs_frac_12m', 'obs_frac_12m_t12', 'obs_frac_12m_t24',
]

MACRO_COLS_CORE = [
    'cpi_all_items_infl_yoy_pct', 'unemployment_rate_sa_pct',
    'sora_compounded_3m_pct', 'sg_govt_bond_yield_10y_pct',
    'sgd_per_usd_logret_12m_pct', 'ura_private_rental_index_yoy_pct',
    'ura_private_price_index_avg_3seg_yoy_pct',
    'hdb_resale_price_index_yoy_pct', 'gva_yoy_growth_pct',
]

# Applied in fill_missing step 8 to MACRO_COLS_CORE only (see module docstring).
# ── Release-lag assumptions (Singapore, conservative) ─────────────────────────
# CPI                        : published ~3 weeks after ref month  → 2m shift safe
# SORA / bond yield / FX     : daily market data, real-time        → 2m very conservative
# Unemployment                : MOM quarterly, ~8 weeks after qtr  → 2m borderline; see note
# GVA/GDP                    : MTI advance ~6w, revised ~12w       → needs 3m minimum
# URA rental/price index     : quarterly, ~4 weeks after qtr end   → lag1q already applied upstream
# HDB resale price index     : quarterly, ~4 weeks after qtr end   → 2m borderline
#
# Decision: raise CORE_MACRO_EXTRA_PANEL_SHIFT_MONTHS from 2 → 3 to cover
# GVA revised release and quarterly series with uncertain pub dates.
# This is still a modelling assumption, not a verified vintage join.
# For full audit, build a release-calendar table and join by last_pub_date <= origin.
# ─────────────────────────────────────────────────────────────────────────────
CORE_MACRO_EXTRA_PANEL_SHIFT_MONTHS = 3  # raised from 2; see release-lag note above

MACRO_COLS_EXTENDED = [
    'rental_index_nonlanded_yoy_log_growth_lag1q',
    'rental_index_nonlanded_in_ccr_yoy_log_growth_lag1q',
    'rental_index_nonlanded_in_ocr_yoy_log_growth_lag1q',
    'rental_index_nonlanded_in_rcr_yoy_log_growth_lag1q',
    'ppi_nonlanded_yoy_log_growth_lag1q',
    'ppi_nonlanded_in_ccr_yoy_log_growth_lag1q',
    'ppi_nonlanded_in_ocr_yoy_log_growth_lag1q',
    'ppi_nonlanded_in_rcr_yoy_log_growth_lag1q',
    'log1p_vacant_nonlanded_units_lag1q',
    'vacancy_rate_nonlanded_units_lag1q',
    'log1p_nonlanded_private_units_in_the_pipeline_lag1q',
    'nonlanded_private_units_in_the_pipeline_under_construction_share_lag1q',
    'log1p_non_landed_units_launched_in_private_res_projects_with_prereq_for_sale_lag1q',
    'non_landed_units_in_private_res_projects_with_prereq_not_launched_share_lag1q',
    'log1p_unsold_completed_nonlanded_private_units_lag1q',
    'log1p_unsold_non_landed_units_in_launched_private_res_projects_lag1q',
    'assessed_non_landed_private_residential_owned_by_companies_share_lag1q',
    'assessed_non_landed_private_residential_owned_by_pr_foreigners_share_lag1q',
    'uncompleted_non_landed_private_residential_purchased_by_companies_share_lag1q',
    'uncompleted_non_landed_private_residential_purchased_by_foreigners_share_lag1q',
    'uncompleted_non_landed_private_residential_purchased_by_spr_share_lag1q',
    # Supply features v2 (Developer Sales + Project Details)
    'project_sold_share_lag1m',
    'project_sold_share_mom_chg_lag1m',
    'district_units_launched_lag1m',
    'district_units_under_construction_qoq_lag1q',
    'district_units_not_launched_yet_qoq_lag1q',
    'district_units_launched_unsold_qoq_lag1q',
    'district_units_completed_unsold_qoq_lag1q',
    # MOM features v2
    'retrenchment_yoy_pct_lag1q',
    'labour_churn_index_lag1q',
    'median_income_yoy_pct_lag',
    'ep_share_yoy_pct_lag1y',

    # ── Trend & Policy features (added post quarterly-meeting) ─────────────
    'gt_condo_rental_singapore_ma3',
    'gt_hdb_rent_singapore_ma3',
    'gt_singapore_international_schools_ma3',
    'gt_cost_of_living_singapore_ma3',
    'absd_investor_avg_level',
    'absd_event_3m',
    'resi_composite_index',
    'resi_future_index',
    'gls_resi_num_sites_awarded_36m',
    'gls_resi_total_gfa_awarded_36m',
    # Sales-specific
    'gt_condo_sale_ma3',
    'gt_new_condo_launch_singapore_ma3',
    'gt_hdb_resale_singapore_ma3',
    'gt_en_bloc_singapore_ma3',
    'gls_resi_median_land_price_psm_gfa_12m',
    'gls_resi_avg_num_bids_12m',
]

MACRO_COLS = MACRO_COLS_CORE + MACRO_COLS_EXTENDED

DISCOUNT_LAG1Q_COL = 'new_resale_discount_lag1q'

AVG_NEW_SALE_PSF_LAG1Q_COL = 'avg_new_sale_psf_lag1q'
AVG_RESALE_PSF_LAG1Q_COL   = 'avg_resale_psf_lag1q'

# Raw contemporaneous monthly Planning-Area averages.  These must retain NaN
# for no-transaction months until calendar rolling/lag features are built.
AVG_NEW_SALE_PSF_RAW_COL = 'avg_new_sale_psf'
AVG_RESALE_PSF_RAW_COL   = 'avg_resale_psf'

PROTECTED_COLS = {
    'rent_psf_obs', 'rent_psf_imp', 'was_observed',
    'target_t1', 'target_t12', 'target_t24',
    'consulting_rental_forecast_mid',
    'consulting_rental_direction',
    'consulting_sale_forecast_mid',
    'consulting_sale_direction',
    AVG_NEW_SALE_PSF_RAW_COL,
    AVG_RESALE_PSF_RAW_COL,
}


# ── Load ──────────────────────────────────────────────────────────────────────

def load_panel(csv_path: str) -> pd.DataFrame:
    """Load the VAE-imputed panel CSV, add period / month_of_year / lease_year."""
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values(['project_idx', 'date']).reset_index(drop=True)

    df['period']        = df['date'].dt.year * 100 + df['date'].dt.month
    df['month_of_year'] = df['date'].dt.month
    df['lease_year']    = df['date'].dt.year

    obs_frac_cols = [c for c in DROP_MODEL_COLS if c.startswith('obs_frac_') and c in df.columns]
    if obs_frac_cols:
        df.drop(columns=obs_frac_cols, inplace=True)

    print(f'Loaded   : {len(df):,} rows x {len(df.columns)} columns')
    print(f'Projects : {df["project_idx"].nunique():,}')
    print(f'Period   : {df["period"].min()} → {df["period"].max()}')
    return df


# ── Fill missing values ───────────────────────────────────────────────────────

def _fill_planning_area_lag1q_col(
    df: pd.DataFrame,
    col: str,
    train_mask: pd.Series,
) -> None:
    """
    In-place: fill NaNs in a Planning-Area-specific, already-lag1q-at-source
    column using the per-Planning-Area **train-period** median, falling back
    to the global train-period median. No-leakage (train_mask only).

    Shared helper for DISCOUNT_LAG1Q_COL, AVG_NEW_SALE_PSF_LAG1Q_COL, and
    AVG_RESALE_PSF_LAG1Q_COL — all three are merged the same way upstream
    (Planning Area x quarter, lagged 1 quarter at source) and so share the
    same fallback policy.
    """
    if col not in df.columns or 'Planning Area' not in df.columns:
        return

    train_area_median = (
        df.loc[train_mask]
          .groupby('Planning Area')[col]
          .median()
    )
    mapped = df['Planning Area'].map(train_area_median)
    df[col] = df[col].fillna(mapped)

    global_train_median = df.loc[train_mask, col].median()
    if pd.isna(global_train_median):
        global_train_median = 0.0
    df[col] = df[col].fillna(global_train_median)


def fill_missing(
    df: pd.DataFrame,
    test_cutoff: int,
    protected_cols: set | None = None,
) -> pd.DataFrame:
    """
    Impute missing values strictly preventing P1 data leakage:
    - All medians are estimated only using the training split (period < test_cutoff).
    - The clean training parameters are then applied to fill the entire dataset.
    """
    if protected_cols is None:
        protected_cols = PROTECTED_COLS
    
    # Isolate training split to compute parameters without looking ahead into validation/test data
    train_mask = df['period'] < test_cutoff
    df_train = df.loc[train_mask]

    # 1. Amenity distances (exclude MRT and school — handled separately)
    amenity_dist_cols = [
        c for c in df.columns
        if c.startswith('nearest_') and c not in
        ['nearest_mrt_dist', 'nearest_mrt_name', 'nearest_school_dist', 'nearest_school_name']
    ]
    for col in amenity_dist_cols:
        train_median = df_train[col].median()
        fallback = 0.0 if pd.isna(train_median) else train_median
        df[col] = df[col].fillna(fallback)

    # 2. Count columns (Constant fill - naturally leak-free)
    for col in [c for c in df.columns if c.endswith('_top3')]:
        df[col] = df[col].fillna(0)

    # 3. School distances and Has_School* flags
    school_dist_cols = [c for c in df.columns if 'School' in c or 'Junior' in c or 'Institution' in c]
    for col in school_dist_cols + ['nearest_school_dist']:
        if col in df.columns:
            train_median = df_train[col].median()
            fallback = 0.0 if pd.isna(train_median) else train_median
            df[col] = df[col].fillna(fallback)
    if 'nearest_school_name' in df.columns:
        df['nearest_school_name'] = df['nearest_school_name'].fillna('Unknown')

    # 4. Lat / lon
    for col in ['latitude', 'longitude']:
        if col in df.columns:
            train_median = df_train[col].median()
            fallback = 1.35208 if pd.isna(train_median) else train_median  # SG default coordinate fallback
            df[col] = df[col].fillna(fallback)

    # 5. condo_age_at_transaction: district median from training data -> global train median fallback
    train_age_median = df_train['condo_age_at_transaction'].median()
    if pd.isna(train_age_median):
        train_age_median = 0.0
        
    # Group and compute medians ONLY on the training subset to avoid district historical lookahead
    district_medians = df_train.groupby('district')['condo_age_at_transaction'].median()
    mapped_medians = df['district'].map(district_medians)
    
    df['condo_age_at_transaction'] = df['condo_age_at_transaction'].fillna(mapped_medians)
    df['condo_age_at_transaction'] = df['condo_age_at_transaction'].fillna(train_age_median)

    # 6. tenure_remaining_years: train-period median
    train_tenure_median = df_train['tenure_remaining_years'].median()
    if pd.isna(train_tenure_median):
        train_tenure_median = 99.0  # Common default for leasehold properties in SG
    df['tenure_remaining_years'] = df['tenure_remaining_years'].fillna(train_tenure_median)

    # 7. Macro columns: ffill in time order, then train-period median (No-Leakage)
    macro_cols = [c for c in MACRO_COLS if c in df.columns]
    macro_by_date = (
        df.sort_values('date')
          .groupby('date', as_index=False)[macro_cols]
          .first()
          .sort_values('date')
    )
    macro_by_date[macro_cols] = macro_by_date[macro_cols].ffill()
    df = df.drop(columns=macro_cols).merge(macro_by_date, on='date', how='left')

    train_macro_median = df.loc[train_mask, macro_cols].median()
    train_macro_median = train_macro_median.fillna(0.0)
    df[macro_cols] = df[macro_cols].fillna(train_macro_median)

    # 8. Core macros: extra lag by calendar month on lease period
    core_macro_present = [c for c in MACRO_COLS_CORE if c in df.columns]
    k = CORE_MACRO_EXTRA_PANEL_SHIFT_MONTHS
    _lag_macros_by_calendar_month_inplace(df, core_macro_present, k)
    for col in core_macro_present:
        fallback_val = train_macro_median[col] if col in train_macro_median.index else df_train[col].median()
        if pd.isna(fallback_val):
            fallback_val = 0.0
        df[col] = df[col].fillna(fallback_val)
        
    print(f'Core macros (MACRO_COLS_CORE): calendar-month lag k={k} on lease period. Columns: {core_macro_present}')

    # 9. Binary project flags
    df['Large_Dev_200plus'] = df['Large_Dev_200plus'].fillna(0)
    for col in ['project_size_small', 'project_size_medium', 'project_size_large']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 10. MRT distance + categorical string columns
    train_mrt_median = df_train['nearest_mrt_dist'].median()
    if pd.isna(train_mrt_median):
        train_mrt_median = 500.0
    df['nearest_mrt_dist'] = df['nearest_mrt_dist'].fillna(train_mrt_median)
    
    for col in ['Planning Area', 'Neighbourhood', 'condo_name', 'size_tier',
                'size_label', 'segment', 'nearest_mrt_name']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # 11. Planning-Area-specific, already-lag1q-at-source columns: train-period
    # per-Planning-Area median -> global train median fallback (no-leakage).
    # Covers DISCOUNT_LAG1Q_COL (older CSVs) and the two avg-PSF columns
    # (newer CSVs) — whichever of the three are present in df.
    for col in (DISCOUNT_LAG1Q_COL, AVG_NEW_SALE_PSF_LAG1Q_COL, AVG_RESALE_PSF_LAG1Q_COL):
        _fill_planning_area_lag1q_col(df, col, train_mask)

    # 12. Blanket median fill for remaining numeric columns (strictly using Train medians)
    for col in df.select_dtypes(include='number').columns:
        if col not in protected_cols and df[col].isna().sum() > 0:
            train_col_median = df_train[col].median() if col in df_train.columns else np.nan
            fallback = 0.0 if pd.isna(train_col_median) else train_col_median
            df[col] = df[col].fillna(fallback)

    miss = df.isna().sum()
    remaining = miss[miss > 0].to_dict()
    print('Missing after fill:', remaining if remaining else 'None')
    return df
