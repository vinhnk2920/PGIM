"""Rental features from the main full-district notebook; no model fitting on import."""


def prepare(settings):
    from functions.forecasting.memory import downscale_numeric_frame
    import numpy as np
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
    USE_OBS_LAGS_H12 = True
    USE_OBS_LAGS_H24 = True
    H24_NOIMP_CORE_MACROS_ONLY = True
    CSV_PATH = settings.read_model_input("rental")
    settings.read_bucket_ranges()
    TARGET_MODE = "level"
    TEST_CUTOFF = settings.test_cutoff
    print("[rental forecasting features] phase 4")
    df = load_panel(CSV_PATH)
    missing_nearby = sorted(set(NEARBY_PRICE_FEATURES) - set(df.columns))
    if missing_nearby:
        raise KeyError(f"Missing nearby-price features: {missing_nearby}")
    df = downscale_numeric_frame(df)
    print("Nearby-price feature missing rates:")
    print(df[NEARBY_PRICE_FEATURES].isna().mean().sort_values().to_string())
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
    print("[rental forecasting features] phase 5")
    bad_condo_name = df["condo_name"].isna() | df["condo_name"].astype("string").str.strip().isin(
        ["", "Unknown"]
    )
    if bad_condo_name.any():
        examples = df.loc[bad_condo_name, ["project_idx", "period"]].head(10)
        raise ValueError(
            f"Full-district input has {int(bad_condo_name.sum()):,} missing/Unknown condo_name rows. Fix the upstream full-district mapping; do not use the old CCR-only fallback.\nExamples:\n{examples.to_string(index=False)}"
        )
    print(f"condo_name OK: {df['condo_name'].nunique():,} projects across full-district input")
    print("[rental forecasting features] phase 6")
    df = fill_missing(df, TEST_CUTOFF, protected_cols=PROTECTED_COLS | set(NEARBY_VALUE_FEATURES))
    print("[rental forecasting features] phase 7")
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
    df["log_growth_1m_imp"] = df.groupby(_lag_group_keys, dropna=False)["log_rent_psf_imp"].diff()
    df["log_growth_1m_obs"] = df.groupby(_lag_group_keys, dropna=False)["log_rent_psf_obs"].diff()
    lag_source_imp = "log_growth_1m_imp" if TARGET_MODE == "log_growth" else "rent_psf_imp"
    lag_source_obs = "log_growth_1m_obs" if TARGET_MODE == "log_growth" else "rent_psf_obs"
    df = build_lag_features(df, source_col=lag_source_imp)
    df = build_obs_lag_features(df, source_col=lag_source_obs)
    df, CAT_FEATURES = encode_categoricals(df)
    df = downscale_numeric_frame(df)
    print("[rental forecasting features] phase 9")
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
            "h24_noimp_core_macros_only": H24_NOIMP_CORE_MACROS_ONLY,
        },
        structural_features=list(dict.fromkeys(BASE_STRUCTURAL + NEARBY_PRICE_FEATURES)),
    )
    FL["FEATURES_A"]
    FL["CAT_FEATURES_A"]
    FL["FEATURES_OBS"]
    FL["CAT_FEATURES_OBS"]
    FL["FEATURES_T12"]
    FL["FEATURES_T12_OBS"]
    FL["FEATURES_T24"]
    FL["FEATURES_T24_OBS"]
    return (df, FL, CAT_FEATURES, _SIZE_TIER_BY_PROJECT)
