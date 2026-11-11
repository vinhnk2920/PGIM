"""Sales features from the main full-district notebook; no model fitting on import."""


def prepare(settings):
    from functions.forecasting.memory import downscale_numeric_frame
    import numpy as np
    import pandas as pd
    from functions.forecasting.data_loader import (
        load_panel,
        fill_missing,
        DROP_MODEL_COLS,
        PROTECTED_COLS,
    )
    from functions.forecasting.feature_engineering import (
        build_lag_features,
        build_obs_lag_features,
        encode_categoricals,
        make_feature_lists,
        add_post_covid_dummy,
        BASE_STRUCTURAL,
    )
    from functions.common import add_months

    NEARBY_PRICE_FEATURES = [
        "nearby_500m_median_psf_6m_lag1m",
        "nearby_500m_median_psf_12m_lag1m",
        "nearby_500m_observed_condos_12m_lag1m",
        "nearby_500m_known_condos_lag1m",
        "nearby_500m_median_psf_6m_log_growth_12m_lag1m",
        "nearby_1000m_median_psf_6m_lag1m",
        "nearby_1000m_median_psf_12m_lag1m",
        "nearby_1000m_observed_condos_12m_lag1m",
        "nearby_1000m_known_condos_lag1m",
        "nearby_1000m_median_psf_6m_log_growth_12m_lag1m",
    ]
    NEARBY_COUNT_FEATURES = [
        c for c in NEARBY_PRICE_FEATURES if "observed_condos" in c or "known_condos" in c
    ]
    NEARBY_VALUE_FEATURES = [c for c in NEARBY_PRICE_FEATURES if c not in NEARBY_COUNT_FEATURES]
    USE_OBS_LAGS_H1 = False
    USE_OBS_LAGS_H12 = False
    USE_OBS_LAGS_H24 = False
    H1_NOIMP_CORE_MACROS_ONLY = False
    H12_NOIMP_CORE_MACROS_ONLY = False
    H24_NOIMP_CORE_MACROS_ONLY = False
    CSV_PATH = settings.read_model_input("sales")
    SIZE_BUCKET_RANGES_PATH = settings.read_bucket_ranges()
    TARGET_MODE = "log_growth"
    TEST_CUTOFF = settings.test_cutoff
    from functions.experiment import reported
    if reported(settings):
        from functions.forecasting.legacy_data_loader import fill_missing
    print("[sales forecasting features] phase 5")
    if not SIZE_BUCKET_RANGES_PATH.exists():
        raise FileNotFoundError(f"Size bucket range file not found: {SIZE_BUCKET_RANGES_PATH}")
    df = load_panel(CSV_PATH)
    rename_map = {
        "sale_psf_obs": "rent_psf_obs",
        "sale_psf_imp": "rent_psf_imp",
        "sale_psf_mu": "rent_psf_mu",
        "sale_psf_pi_lo": "rent_psf_pi_lo",
        "sale_psf_pi_hi": "rent_psf_pi_hi",
    }
    present = {k: v for k, v in rename_map.items() if k in df.columns}
    missing_core = [c for c in ["sale_psf_obs", "sale_psf_imp"] if c not in df.columns]
    if missing_core:
        raise KeyError(f"Missing required sales columns: {missing_core}")
    if present:
        df = df.rename(columns=present)
    df = downscale_numeric_frame(df)
    missing_nearby = sorted(set(NEARBY_PRICE_FEATURES) - set(df.columns))
    if missing_nearby:
        raise KeyError(f"Missing nearby-price features: {missing_nearby}")
    print("Nearby-price feature missing rates:")
    print(df[NEARBY_PRICE_FEATURES].isna().mean().sort_values().to_string())
    print(f"Loaded size bucket ranges -> {SIZE_BUCKET_RANGES_PATH}")
    print("was_observed distribution:")
    print(
        df["was_observed"].value_counts(normalize=True).rename({True: "Observed", False: "Imputed"})
    )
    _size_tier_pairs = df[["project_idx", "size_tier"]].drop_duplicates()
    _size_tier_conflicts = _size_tier_pairs.groupby("project_idx")["size_tier"].nunique()
    if (_size_tier_conflicts > 1).any():
        raise ValueError("A project_idx maps to more than one size_tier")
    _SIZE_TIER_BY_PROJECT = _size_tier_pairs.set_index("project_idx")["size_tier"]
    if _SIZE_TIER_BY_PROJECT.isna().any():
        raise ValueError("Missing size_tier in the project mapping")
    print("[sales forecasting features] phase 6")
    bad_condo_name = df["condo_name"].isna() | df["condo_name"].astype("string").str.strip().isin(
        ["", "Unknown"]
    )
    if bad_condo_name.any():
        examples = df.loc[bad_condo_name, ["project_idx", "period"]].head(10)
        raise ValueError(
            f"Full-district input has {int(bad_condo_name.sum()):,} missing/Unknown condo_name rows. Fix the upstream mapping; the old CCR-only fallback is disabled.\nExamples:\n{examples.to_string(index=False)}"
        )
    print(f"condo_name OK: {df['condo_name'].nunique():,} projects")
    print("[sales forecasting features] phase 7")
    df = fill_missing(df, TEST_CUTOFF, protected_cols=PROTECTED_COLS | set(NEARBY_VALUE_FEATURES))
    print("[sales forecasting features] phase 8")
    df = add_post_covid_dummy(df)
    EPS = 1e-09
    df["log_rent_psf_imp"] = np.log(df["rent_psf_imp"].clip(lower=EPS))
    df["log_rent_psf_obs"] = np.log(df["rent_psf_obs"].clip(lower=EPS))
    _lag_group_keys = [c for c in ["project_idx", "size_tier"] if c in df.columns]
    if not _lag_group_keys:
        raise KeyError(
            "Need at least one grouping key from ['project_idx', 'size_tier'] for log growth lags"
        )
    df = df.sort_values(_lag_group_keys + ["period"]).copy()
    PRICE_CONTEXT_FEATURES = [
        "log_price_lag_1m_imp",
        "log_price_vs_global_lag_1m",
        "log_price_vs_district_lag_1m",
        "log_price_vs_planning_area_lag_1m",
        "log_price_vs_own_prior12m_lag_1m",
    ]
    _district_context_col = "Postal District" if "Postal District" in df.columns else "district"
    _planning_context_col = "Planning Area" if "Planning Area" in df.columns else "planning_area"
    for _required_context_col in [_district_context_col, _planning_context_col]:
        if _required_context_col not in df.columns:
            raise KeyError(
                f"Cannot build lagged price context; missing grouping column {_required_context_col!r}"
            )
    _context_lag_1m = df[
        _lag_group_keys + ["period", _district_context_col, _planning_context_col]
    ].copy()
    _context_lag_1m["log_price_lag_1m_imp"] = df["log_rent_psf_imp"].to_numpy()
    _context_lag_1m["log_price_vs_global_lag_1m"] = (
        df["log_rent_psf_imp"]
        - df.groupby("period", dropna=False)["log_rent_psf_imp"].transform("median")
    ).to_numpy()
    _context_lag_1m["log_price_vs_district_lag_1m"] = (
        df["log_rent_psf_imp"]
        - df.groupby(["period", _district_context_col], dropna=False)["log_rent_psf_imp"].transform(
            "median"
        )
    ).to_numpy()
    _context_lag_1m["log_price_vs_planning_area_lag_1m"] = (
        df["log_rent_psf_imp"]
        - df.groupby(["period", _planning_context_col], dropna=False)["log_rent_psf_imp"].transform(
            "median"
        )
    ).to_numpy()
    _own_prior12m_log_median = df.groupby(_lag_group_keys, dropna=False, observed=False)[
        "log_rent_psf_imp"
    ].transform(lambda s: s.shift(1).rolling(window=12, min_periods=6).median())
    _context_lag_1m["log_price_vs_own_prior12m_lag_1m"] = (
        df["log_rent_psf_imp"] - _own_prior12m_log_median
    ).to_numpy()
    _context_key_cols = _lag_group_keys + ["period"]
    _context_lag_1m = (
        _context_lag_1m[_context_key_cols + PRICE_CONTEXT_FEATURES]
        .groupby(_context_key_cols, as_index=False, dropna=False, observed=False)[
            PRICE_CONTEXT_FEATURES
        ]
        .median()
    )
    _context_lag_1m["period"] = _context_lag_1m["period"].map(lambda p: add_months(int(p), 1))
    df = df.merge(_context_lag_1m, on=_context_key_cols, how="left", validate="many_to_one")
    df = df.sort_values(_lag_group_keys + ["period"]).copy()
    df["log_price_vs_own_prior12m_lag_1m"] = df["log_price_vs_own_prior12m_lag_1m"].fillna(0.0)
    _context_values = df[PRICE_CONTEXT_FEATURES].to_numpy(dtype=float, na_value=np.nan)
    _context_inf_counts = pd.Series(
        np.isinf(_context_values).sum(axis=0), index=PRICE_CONTEXT_FEATURES
    )
    assert (_context_inf_counts == 0).all(), (
        f"Infinite values found in lagged price-context features: {_context_inf_counts[_context_inf_counts > 0].to_dict()}"
    )
    print("Lagged price-context features created (calendar t-1):", PRICE_CONTEXT_FEATURES)
    print("Lagged price-context missing rates:")
    print(df[PRICE_CONTEXT_FEATURES].isna().mean().to_string())
    df["log_growth_1m_imp"] = df.groupby(_lag_group_keys, dropna=False)["log_rent_psf_imp"].diff()
    df["log_growth_1m_obs"] = df.groupby(_lag_group_keys, dropna=False)["log_rent_psf_obs"].diff()
    lag_source_imp = "log_growth_1m_imp" if TARGET_MODE == "log_growth" else "rent_psf_imp"
    lag_source_obs = "log_growth_1m_obs" if TARGET_MODE == "log_growth" else "rent_psf_obs"
    df = build_lag_features(df, source_col=lag_source_imp)
    df = build_obs_lag_features(df, source_col=lag_source_obs)
    df, CAT_FEATURES = encode_categoricals(df)
    assert lag_source_imp == "log_growth_1m_imp"
    assert lag_source_obs == "log_growth_1m_obs"
    print("Verified: sale lag families are built from one-month log growth.")
    df = downscale_numeric_frame(df)
    print("[sales forecasting features] phase 9")
    EXPORT_META_COLS = [
        "project_idx",
        "condo_name",
        "district",
        "segment",
        "date",
        "period",
        "size_tier",
        "rent_psf_obs",
        "rent_psf_imp",
        "was_observed",
    ]
    EXPORT_META_COLS = [c for c in EXPORT_META_COLS if c in df.columns]
    df_export_meta = df[EXPORT_META_COLS].copy()
    print(f"Export metadata snapshot: {df_export_meta.shape}")
    print("[sales forecasting features] phase 10")
    drop_model = [c for c in DROP_MODEL_COLS if c in df.columns]
    if drop_model:
        df.drop(columns=drop_model, inplace=True)
        CAT_FEATURES = [c for c in CAT_FEATURES if c in df.columns]
        print(f"Removed from modelling df: {drop_model}")
    FL = make_feature_lists(
        df,
        CAT_FEATURES,
        flags={
            "use_obs_h1": USE_OBS_LAGS_H1,
            "use_obs_h12": USE_OBS_LAGS_H12,
            "use_obs_h24": USE_OBS_LAGS_H24,
            "h1_noimp_core_macros_only": H1_NOIMP_CORE_MACROS_ONLY,
            "h12_noimp_core_macros_only": H12_NOIMP_CORE_MACROS_ONLY,
            "h24_noimp_core_macros_only": H24_NOIMP_CORE_MACROS_ONLY,
        },
        structural_features=list(dict.fromkeys(BASE_STRUCTURAL + NEARBY_PRICE_FEATURES)),
    )
    FEATURES_A = FL["FEATURES_A"]
    FL["CAT_FEATURES_A"]
    FEATURES_OBS = FL["FEATURES_OBS"]
    FL["CAT_FEATURES_OBS"]
    FEATURES_T12 = FL["FEATURES_T12"]
    FEATURES_T12_OBS = FL["FEATURES_T12_OBS"]
    FEATURES_T24 = FL["FEATURES_T24"]
    FEATURES_T24_OBS = FL["FEATURES_T24_OBS"]
    _price_context_live = [c for c in PRICE_CONTEXT_FEATURES if c in df.columns]
    if len(_price_context_live) != len(PRICE_CONTEXT_FEATURES):
        _missing_context = sorted(set(PRICE_CONTEXT_FEATURES) - set(_price_context_live))
        raise KeyError(f"Missing Experiment 2 price-context features: {_missing_context}")
    FEATURES_A = list(dict.fromkeys(FEATURES_A + _price_context_live))
    FEATURES_T12 = list(dict.fromkeys(FEATURES_T12 + _price_context_live))
    FEATURES_T24 = list(dict.fromkeys(FEATURES_T24 + _price_context_live))
    assert not set(_price_context_live) & set(FEATURES_OBS)
    assert not set(_price_context_live) & set(FEATURES_T12_OBS)
    assert not set(_price_context_live) & set(FEATURES_T24_OBS)
    print(f"Experiment 2 added to Imputed models only: {_price_context_live}")
    print(
        f"V2 feature counts — h1={len(FEATURES_A)}, h12={len(FEATURES_T12)}, h24={len(FEATURES_T24)}"
    )
    FL.update(FEATURES_A=FEATURES_A, FEATURES_T12=FEATURES_T12, FEATURES_T24=FEATURES_T24)
    return (df, FL, CAT_FEATURES, _SIZE_TIER_BY_PROJECT)
