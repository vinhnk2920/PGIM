"""Sales full-district VAE orchestration. Shared algorithms are in core.py; plotting-only cells removed."""


def run(settings):
    from functions.experiment import profile, reported, record_panel
    from functions.imputation.artifacts import artifact_paths, run_timestamp, save_artifacts
    from functions.imputation.schema import (
        AMENITY_TYPES,
        LOCATION_COLS,
        MACRO_COLS,
        ORDINALS,
        PROPERTY_COLS,
    )
    from types import SimpleNamespace
    from functions.imputation.core import (
        build_context_matrix,
        clean_num,
        conformal_calibrate_v6,
        evaluate,
        export_bucket_ranges,
        infer_vae_a4,
        make_size_bucket,
        train_vae_a4,
        val_rmse_a4,
        validate_bucket_ranges,
    )

    print("[sales imputation] source phase 2")
    import os, time, re
    import numpy as np
    import pandas as pd
    from itertools import product as iproduct
    import torch
    from tqdm.auto import tqdm

    SEED = settings.seed
    settings.folder("02_imputation")
    DATA_PATH = settings.read_prepared("sales")
    N_SIZE_BUCKETS = 5
    SIZE_BUCKET_RANGES_CSV = settings.read_bucket_ranges()
    ALL_DISTRICTS = list(range(1, 29))
    EXPECTED_DISTRICTS = set(ALL_DISTRICTS) - {24}
    N_MC_DRAWS = settings.mc_draws
    AMENITY_DIST_THRESHOLD = 500.0
    started_at = run_timestamp()
    OUT_DIR = artifact_paths(settings, "sales")["folder"]
    CSV_OUT = str(settings.imputed("sales"))
    PSF_REASONABLE_RANGE = (300.0, 3500.0)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True
    print(f"PyTorch {torch.__version__} | Device: {DEVICE}")
    print(f"Output dir : {OUT_DIR}")
    print(f"CSV  : {CSV_OUT}")
    print(f"Amenity threshold : {AMENITY_DIST_THRESHOLD}m")
    print("[sales imputation] source phase 3")
    print("Helper functions defined.")
    print("[sales imputation] source phase 4")
    all_file_cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    mrt_cols = [c for c in all_file_cols if c.endswith("_MRT")]
    has_cols = [c for c in all_file_cols if c.startswith("Has_")]
    standard_prefixes = (
        "Has_",
        "Condo_",
        "tenure_",
        "project_size_",
        "Large_",
        "Planning",
        "Neighbourhood",
        "Floor",
        "Monthly",
        "Lease",
        "Property",
        "Project",
        "Area",
        "Sale",
        "Transacted",
        "Unit Price",
        "Type of",
        "Nett",
        "Number",
        "Completion",
        "Purchaser",
        "Address",
        "Tenure",
        "index",
        "Postal",
        "Street",
        "No of",
        "onemap",
        "latitude",
        "longitude",
        "house_",
        "district",
        "Planning Region",
    )
    macro_and_index = set(MACRO_COLS)
    school_cols = [
        c
        for c in all_file_cols
        if not c.endswith("_MRT")
        and (not c.startswith("Has_"))
        and (not any((c.startswith(p) for p in standard_prefixes)))
        and (c not in macro_and_index)
        and (not re.match("^\\d", c))
        and (c not in ["district"])
        and pd.Series([c]).str.contains("School|Junior|College|Institution", case=False).any()
    ]
    amenity_dist_cols = [
        f"{o} {a} dist" for o in ORDINALS for a in AMENITY_TYPES if f"{o} {a} dist" in all_file_cols
    ]
    amenity_name_cols = [
        f"{o} {a}" for o in ORDINALS for a in AMENITY_TYPES if f"{o} {a}" in all_file_cols
    ]
    loc_cols_avail = [c for c in LOCATION_COLS if c in all_file_cols]
    base_load = [
        "Project Name",
        "Property Type",
        "Postal District",
        "Area (SQFT)",
        "Unit Price ($ PSF)",
        "Sale Date",
        "Transacted Price ($)",
    ]
    # Age fallback fix (2026-10-28): load this sales-only source field even
    # though it is intentionally excluded from the shared exported PROPERTY_COLS.
    load_cols = (
        base_load
        + [c for c in PROPERTY_COLS if c in all_file_cols]
        + (["house_age_2026"] if "house_age_2026" in all_file_cols else [])
        + [c for c in MACRO_COLS if c in all_file_cols]
        + mrt_cols
        + has_cols
        + school_cols
        + amenity_dist_cols
        + amenity_name_cols
        + loc_cols_avail
    )
    load_cols = list(dict.fromkeys(load_cols))
    print(f"Columns to load   : {len(load_cols)}")
    print(f"  MRT cols        : {len(mrt_cols)}")
    print(f"  Has_* cols      : {len(has_cols)}")
    print(f"  School cols     : {len(school_cols)}")
    print(f"  Amenity dist    : {len(amenity_dist_cols)}")
    print(f"  Amenity name    : {len(amenity_name_cols)}")
    print(f"  Location cols   : {loc_cols_avail}")
    print("[sales imputation] source phase 5")
    print(f"Loading from {DATA_PATH} ...")
    df_raw = pd.read_csv(DATA_PATH, usecols=load_cols)
    df_raw["Postal District"] = pd.to_numeric(df_raw["Postal District"], errors="coerce").astype(
        "Int64"
    )
    df_raw = df_raw[df_raw["Postal District"].between(1, 28)].copy()
    loaded_districts = set(df_raw["Postal District"].dropna().astype(int).unique())
    print(f"Districts loaded : {sorted(loaded_districts)}")
    print(f"District count   : {len(loaded_districts)}")
    assert loaded_districts == EXPECTED_DISTRICTS, (
        f"Unexpected district coverage. Missing={sorted(EXPECTED_DISTRICTS - loaded_districts)}, extra={sorted(loaded_districts - EXPECTED_DISTRICTS)}"
    )
    # Fill the canonical age before project/size aggregation and transaction-age
    # calculation. Existing ages win; house_age_2026 remains an input-only field.
    if "Condo_Age_2026" in df_raw.columns and "house_age_2026" in df_raw.columns:
        _age_before = int(df_raw["Condo_Age_2026"].isna().sum())
        df_raw["Condo_Age_2026"] = df_raw["Condo_Age_2026"].fillna(df_raw["house_age_2026"])
        print(
            f"Condo_Age_2026 fallback filled: {_age_before - df_raw['Condo_Age_2026'].isna().sum():,}"
        )
    df_raw["Area_Cleaned"] = clean_num(df_raw["Area (SQFT)"])
    df_raw["Sale_PSF"] = clean_num(df_raw["Unit Price ($ PSF)"])
    if reported(settings):
        from functions.experiment import sales_buckets
        bucket_ranges = pd.read_csv(SIZE_BUCKET_RANGES_CSV)
        df_raw["size_bucket"] = sales_buckets(df_raw["Area_Cleaned"], bucket_ranges)
    else:
        validate_bucket_ranges(pd.read_csv(SIZE_BUCKET_RANGES_CSV))
        bucket_ranges = export_bucket_ranges()
        df_raw["size_bucket"] = make_size_bucket(df_raw["Area_Cleaned"], q=N_SIZE_BUCKETS)
    print(
        f"Validated shared bucket-range reference: {SIZE_BUCKET_RANGES_CSV} ({len(bucket_ranges)} buckets)"
    )
    print("Reported sales cuts from fresh rental quantile bounds" if reported(settings) else
          "Shared fixed size buckets: <=650, (650,950], (950,1150], (1150,1450], >1450 sqft")
    print(
        f"\n{'Bucket':<8} {'Cut lower':>13} {'Cut upper':>13}  {'Observed sales min':>18} {'Observed sales max':>18}  {'N in bucket':>11}"
    )
    print("-" * 85)
    _sales_tmp = df_raw[["Area_Cleaned", "size_bucket"]].rename(
        columns={"Area_Cleaned": "area", "size_bucket": "sz"},
    ).dropna(subset=["area"])
    for _, row in bucket_ranges.iterrows():
        sz = row["size_bucket"]
        r_lo = row["lower_bound"]
        r_hi = row["upper_bound"]
        _snb = _sales_tmp.loc[_sales_tmp["sz"] == sz, "area"]
        s_lo = f"{_snb.min():.0f}" if not _snb.empty else "n/a"
        s_hi = f"{_snb.max():.0f}" if not _snb.empty else "n/a"
        n = int((df_raw["size_bucket"] == sz).sum())
        print(f"{sz:<8} {r_lo:>13.0f} {r_hi:>13.0f}  {s_lo:>18} {s_hi:>18}  {n:>11,}")
    print()
    df_raw["Date_DT"] = pd.to_datetime(df_raw["Sale Date"], errors="coerce")
    df_raw["YearMonth"] = df_raw["Date_DT"].dt.to_period("M").astype(str)
    df_raw["ProjectSeries"] = (
        df_raw["Project Name"].astype(str).str.strip() + " | " + df_raw["size_bucket"]
    )
    valid_mask = df_raw["Sale_PSF"].between(*PSF_REASONABLE_RANGE) & df_raw["Date_DT"].notna()
    df_raw = df_raw[valid_mask].copy()
    all_projects = sorted(df_raw["ProjectSeries"].unique())
    proj_idx_map = {name: i for i, name in enumerate(all_projects)}
    P = len(all_projects)
    print(f"Raw data : {len(df_raw):,} rows | {P} project-size series")
    print("[sales imputation] source phase 6")
    agg = df_raw.groupby(["ProjectSeries", "YearMonth"], as_index=False).agg(
        Sale_PSF_mean=("Sale_PSF", "mean"), n_tx=("Sale_PSF", "size")
    )
    all_months = sorted(agg["YearMonth"].unique())
    T = len(all_months)
    B = 1
    month_idx = {m: j for j, m in enumerate(all_months)}
    X_full = np.full((P, T, B), np.nan, dtype=np.float32)
    M_full = np.zeros((P, T, B), dtype=np.float32)
    for _, row in agg.iterrows():
        p = proj_idx_map[row["ProjectSeries"]]
        t = month_idx[row["YearMonth"]]
        X_full[p, t, 0] = row["Sale_PSF_mean"]
        M_full[p, t, 0] = 1.0
    dates = pd.PeriodIndex(all_months, freq="M").to_timestamp()
    record_panel(settings, "sales", X_full, M_full, all_projects, all_months)
    obs_vals = X_full[M_full == 1]
    print(f"Full panel : P={P}, T={T}, B={B}")
    print(f"Date range : {all_months[0]} → {all_months[-1]}")
    print(f"Observed   : {int(M_full.sum()):,} ({100 * M_full.mean():.1f}%)")
    print(f"PSF range  : [{obs_vals.min():.2f}, {obs_vals.max():.2f}] SGD/sqft")
    print("[sales imputation] source phase 7")
    np.random.seed(SEED)
    obs_pos = [
        (p, t, b) for p in range(P) for t in range(T) for b in range(B) if M_full[p, t, b] == 1
    ]
    np.random.shuffle(obs_pos)
    n_val_es = int(0.05 * len(obs_pos))
    n_cal = int(0.1 * len(obs_pos))
    n_test = int(0.1 * len(obs_pos))
    val_es_pos = obs_pos[:n_val_es]
    cal_pos = obs_pos[n_val_es : n_val_es + n_cal]
    test_pos = obs_pos[n_val_es + n_cal : n_val_es + n_cal + n_test]
    train_pos = obs_pos[n_val_es + n_cal + n_test :]
    M_train = np.zeros((P, T, B), dtype=np.float32)
    for p, t, b in train_pos:
        M_train[p, t, b] = 1.0
    X_train = X_full.copy()
    X_train[M_train == 0] = np.nan
    M_val_es = np.zeros((P, T, B), dtype=np.float32)
    for p, t, b in val_es_pos:
        M_val_es[p, t, b] = 1.0
    M_cal = np.zeros((P, T, B), dtype=np.float32)
    for p, t, b in cal_pos:
        M_cal[p, t, b] = 1.0
    M_test = np.zeros((P, T, B), dtype=np.float32)
    for p, t, b in test_pos:
        M_test[p, t, b] = 1.0
    print(f"Split (full timeline {all_months[0]} → {all_months[-1]}):")
    print(f"  Total obs  : {len(obs_pos):,}")
    print(f"  Train      : {len(train_pos):,}  ({100 * len(train_pos) / len(obs_pos):.0f}%)")
    print(
        f"  Val (ES)   : {len(val_es_pos):,}   ({100 * len(val_es_pos) / len(obs_pos):.0f}%)  <- early stopping only"
    )
    print(
        f"  Cal        : {len(cal_pos):,}  ({100 * len(cal_pos) / len(obs_pos):.0f}%)  <- conformal calibration only"
    )
    print(
        f"  Test       : {len(test_pos):,}  ({100 * len(test_pos) / len(obs_pos):.0f}%)  <- final evaluation"
    )
    print()
    print("Overlap check (must all be 0):")
    val_es_set = set(val_es_pos)
    cal_set = set(cal_pos)
    test_set = set(test_pos)
    train_set = set(train_pos)
    print(f"  train  cap val_es : {len(train_set & val_es_set)}")
    print(f"  train  cap cal    : {len(train_set & cal_set)}")
    print(f"  train  cap test   : {len(train_set & test_set)}")
    print(f"  val_es cap cal    : {len(val_es_set & cal_set)}")
    print(f"  val_es cap test   : {len(val_es_set & test_set)}")
    print(f"  cal    cap test   : {len(cal_set & test_set)}")
    print("[sales imputation] source phase 8")
    C_ctx = build_context_matrix(dates)
    K_CTX = C_ctx.shape[1]
    print(f"Context matrix : {C_ctx.shape}")
    print(f"K_CTX          : {K_CTX}")
    print("[sales imputation] source phase 9")
    print("VAE-A4 architecture defined.")
    context = SimpleNamespace(
        B=B,
        DEVICE=DEVICE,
        K_CTX=K_CTX,
        M_full=M_full,
        M_test=M_test,
        M_train=M_train,
        P=P,
        PSF_REASONABLE_RANGE=PSF_REASONABLE_RANGE,
        T=T,
        X_full=X_full,
        X_train=X_train,
        val_es_pos=val_es_pos,
    )
    print("[sales imputation] source phase 10")
    if reported(settings):
        best_config = dict(profile(settings)["vae_configs"]["sales"])
        best_config["val_rmse"] = None
        print("Fresh VAE training with reported scalar hyperparameters:", best_config)
        gs_df = pd.DataFrame([best_config])
    else:
        a4_grid = {
            "z_dim": [8, 16, 32],
            "beta": [0.01, 0.05, 0.1],
            "lr": [0.001, 0.0003],
            "h": [64, 128, 256],
            "drop": [0.05, 0.1, 0.2],
        }
        configs = list(
            iproduct(a4_grid["z_dim"], a4_grid["beta"], a4_grid["lr"], a4_grid["h"], a4_grid["drop"])
        )
        if settings.imputation_max_configs is not None:
            configs = configs[: settings.imputation_max_configs]
        print(f"Grid config  : {len(configs)} combinations")
        for k, v in a4_grid.items():
            print(f"  {k}: {v}")
        t0_gs = time.time()
        gs_rows = []
        best_config = {"val_rmse": np.inf}
        for z_dim, beta, lr, h, drop in tqdm(configs, desc="VAE-A4 grid"):
            _, rv, e = val_rmse_a4(
                z_dim,
                beta,
                lr,
                C_ctx,
                h=h,
                drop=drop,
                context=context,
                max_epochs=settings.imputation_max_epochs,
            )
            row = dict(z_dim=z_dim, beta=beta, lr=lr, h=h, drop=drop, epochs=e, val_rmse=rv)
            gs_rows.append(row)
            print(f"  z={z_dim} b={beta} lr={lr} h={h} drop={drop} -> epochs={e} val_rmse={rv:.4f}")
            if rv < best_config["val_rmse"]:
                best_config = row.copy()
        print(f"\nDone in {time.time() - t0_gs:.1f}s | Best: {best_config}")
        gs_df = pd.DataFrame(gs_rows).sort_values("val_rmse").reset_index(drop=True)
        print(gs_df.head(10).to_string())
    print("[sales imputation] source phase 11")
    t0_eval = time.time()
    print("Stage 6a: training evaluation model on train split ...")
    eval_vms = train_vae_a4(
        best_config["z_dim"],
        best_config["beta"],
        best_config["epochs"],
        X_full,
        M_full,
        X_train,
        M_train,
        C_ctx,
        lr=best_config["lr"],
        h=best_config["h"],
        drop=best_config["drop"],
        context=context,
    )
    X_eval_input = X_full.copy()
    M_eval_input = M_full.copy()
    for p, t, b in val_es_pos + cal_pos + test_pos:
        X_eval_input[p, t, b] = np.nan
        M_eval_input[p, t, b] = 0.0
    eval_draws_full, _, _ = infer_vae_a4(
        eval_vms, X_eval_input, M_eval_input, C_ctx, n_samples=N_MC_DRAWS, context=context
    )
    X_eval_imputed = np.mean(np.stack(eval_draws_full), axis=0)
    t_eval_elapsed = time.time() - t0_eval
    print(f"Stage 6a done in {t_eval_elapsed:.1f}s")
    t0_train = time.time()
    print("Stage 6b: training production model on full panel ...")
    prod_vms = train_vae_a4(
        best_config["z_dim"],
        best_config["beta"],
        best_config["epochs"],
        X_full,
        M_full,
        X_full,
        M_full,
        C_ctx,
        lr=best_config["lr"],
        h=best_config["h"],
        drop=best_config["drop"],
        context=context,
    )
    prod_draws, X_prod_mu, X_prod_sig = infer_vae_a4(
        prod_vms, X_full, M_full, C_ctx, n_samples=N_MC_DRAWS, context=context
    )
    X_imputed = np.mean(np.stack(prod_draws), axis=0)
    t_train_elapsed = time.time() - t0_train
    print(
        f"Stage 6b done in {t_train_elapsed:.1f}s  |  NaN remaining: {int(np.isnan(X_imputed).sum())}"
    )
    print(f"X_prod_mu  shape: {X_prod_mu.shape}   (P, T, B)")
    print(f"X_prod_sig shape: {X_prod_sig.shape}   (P, T, B)")
    print("[sales imputation] source phase 12")
    t0_conf = time.time()
    q_hat_prod, X_prod_mu2, X_prod_sig2, X_pi_lo, X_pi_hi = conformal_calibrate_v6(
        eval_vms, X_train, M_train, C_ctx, cal_pos, context=context
    )
    print(f"Done in {time.time() - t0_conf:.1f}s")
    print(f"X_pi_lo shape: {X_pi_lo.shape}   (P, T, B)")
    print(f"X_pi_hi shape: {X_pi_hi.shape}   (P, T, B)")
    print("[sales imputation] source phase 13")
    results = evaluate(
        X_eval_imputed,
        "VAE-A4-cs-mcra-v6-eval",
        t_eval_elapsed,
        samples=eval_draws_full,
        context=context,
    )
    cov_list = []
    for p, t, b in test_pos:
        y_i = float(X_full[p, t, b])
        lo_i = float(X_pi_lo[p, t, b])
        hi_i = float(X_pi_hi[p, t, b])
        cov_list.append(lo_i <= y_i <= hi_i)
    coverage = float(np.mean(cov_list) * 100)
    print(f"Conformal PI-90 coverage on test : {coverage:.1f}%  (target >= 90%)")
    print()
    print(f"{'Metric':<10} {'Value':>10}")
    print(f"{'-' * 22}")
    print(f"{'RMSE':<10} {results['RMSE']:>10.4f}")
    print(f"{'MAE':<10} {results['MAE']:>10.4f}")
    print(f"{'MAPE':<10} {results['MAPE']:>9.1f}%")
    print(f"{'R2':<10} {results['R2']:>10.4f}")
    print(f"{'PI-90%':<10} {coverage:>9.1f}%")
    print()
    obs_v = X_full[M_full == 1].ravel()
    imp_v = X_eval_imputed[M_full == 0].ravel()
    imp_v = imp_v[np.isfinite(imp_v)]
    print(f"Observed median : {np.median(obs_v):.2f} SGD/sqft")
    print(f"Imputed  median : {np.median(imp_v):.2f} SGD/sqft")
    print()
    print("NOTE: X_imputed / prod_vms = Stage 6b (full panel) → output CSV only.")
    print("      Test metrics always from Stage 6a (eval model).")
    print("[sales imputation] source phase 14")
    prop_cols_avail = [c for c in PROPERTY_COLS if c in df_raw.columns]
    mrt_cols_avail = [c for c in mrt_cols if c in df_raw.columns]
    has_cols_avail = [c for c in has_cols if c in df_raw.columns]
    school_cols_avail = [c for c in school_cols if c in df_raw.columns]
    loc_cols_avail = [c for c in LOCATION_COLS if c in df_raw.columns]
    macro_cols_avail = [c for c in MACRO_COLS if c in df_raw.columns]
    amenity_dist_avail = [c for c in amenity_dist_cols if c in df_raw.columns]
    [c for c in amenity_name_cols if c in df_raw.columns]
    print(f"Property cols     : {len(prop_cols_avail)}")
    print(f"Has_* cols        : {len(has_cols_avail)}")
    print(f"School cols       : {len(school_cols_avail)}")
    print(f"Location cols     : {loc_cols_avail}")
    print(f"Amenity dist cols : {len(amenity_dist_avail)}")
    print(f"MRT cols          : {len(mrt_cols_avail)}")
    print(f"Macro cols        : {len(macro_cols_avail)}")
    print("[sales imputation] source phase 15")
    base_agg = {
        "condo_name": ("Project Name", lambda x: x.iloc[0].strip()),
        "size_tier": ("size_bucket", lambda x: x.iloc[0]),
        "district": ("Postal District", "median"),
        "area_sqft_med": ("Area_Cleaned", "median"),
        "n_transactions": ("Sale_PSF", "count"),
    }
    series_meta = df_raw.groupby("ProjectSeries").agg(**base_agg).reset_index()
    series_meta["district"] = series_meta["district"].fillna(-1).astype(int)
    series_meta["segment"] = series_meta["district"].apply(
        lambda d: "luxury" if d in [9, 10, 11] else "mass"
    )
    size_label_map = {
        "SZ1": "very_small",
        "SZ2": "small",
        "SZ3": "medium",
        "SZ4": "large",
        "SZ5": "very_large",
    }
    series_meta["size_label"] = series_meta["size_tier"].map(size_label_map)
    series_meta["project_idx"] = series_meta["ProjectSeries"].map(proj_idx_map)
    series_meta = series_meta.dropna(subset=["project_idx"])
    series_meta["project_idx"] = series_meta["project_idx"].astype(int)
    print(f"Base meta: {series_meta.shape}")
    print("[sales imputation] source phase 16")
    if prop_cols_avail:
        num_cols = [c for c in prop_cols_avail if pd.api.types.is_numeric_dtype(df_raw[c])]
        cat_cols = [c for c in prop_cols_avail if not pd.api.types.is_numeric_dtype(df_raw[c])]
        grp = df_raw.groupby("ProjectSeries")
        parts = []
        if num_cols:
            parts.append(grp[num_cols].median())
        if cat_cols:
            parts.append(
                grp[cat_cols].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
            )
        prop_meta = pd.concat(parts, axis=1).reset_index()
        prop_meta["project_idx"] = prop_meta["ProjectSeries"].map(proj_idx_map)
        prop_meta = prop_meta.dropna(subset=["project_idx"])
        prop_meta["project_idx"] = prop_meta["project_idx"].astype(int)
        series_meta = series_meta.merge(
            prop_meta[["project_idx"] + prop_cols_avail], on="project_idx", how="left"
        )
        print(f"After property features: {series_meta.shape}")
    print("[sales imputation] source phase 17")
    if mrt_cols_avail:
        mrt_dist = df_raw[mrt_cols_avail].apply(pd.to_numeric, errors="coerce")
        has_mrt_value = mrt_dist.notna().any(axis=1)
        df_raw["nearest_mrt_dist"] = mrt_dist.min(axis=1)
        nearest_mrt_name = pd.Series(pd.NA, index=df_raw.index, dtype="string")
        nearest_mrt_name.loc[has_mrt_value] = (
            mrt_dist.loc[has_mrt_value]
            .idxmin(axis=1)
            .str.replace("_MRT", "", regex=False)
            .astype("string")
        )
        df_raw["nearest_mrt_name"] = nearest_mrt_name
        print(f"Rows without MRT distance: {(~has_mrt_value).sum():,} / {len(df_raw):,}")
        mrt_meta = (
            df_raw.groupby("ProjectSeries")
            .agg(
                nearest_mrt_dist=("nearest_mrt_dist", "median"),
                nearest_mrt_name=(
                    "nearest_mrt_name",
                    lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA,
                ),
            )
            .reset_index()
        )
        mrt_meta["project_idx"] = mrt_meta["ProjectSeries"].map(proj_idx_map)
        mrt_meta = mrt_meta.dropna(subset=["project_idx"])
        mrt_meta["project_idx"] = mrt_meta["project_idx"].astype(int)
        series_meta = series_meta.merge(
            mrt_meta[["project_idx", "nearest_mrt_dist", "nearest_mrt_name"]],
            on="project_idx",
            how="left",
        )
        print(f"After MRT features: {series_meta.shape}")
    print("[sales imputation] source phase 18")

    def parse_distance_to_metres(series):
        s = series.astype(str).str.strip().str.lower()
        extracted = s.str.extract("([\\d.]+)\\s*(km|m)?")
        values = pd.to_numeric(extracted[0], errors="coerce")
        unit = extracted[1]
        multiplier = unit.map({"km": 1000.0, "m": 1.0}).fillna(1.0)
        return values * multiplier

    if amenity_dist_avail:
        all_amenity_dist = [
            f"{o} {a} dist"
            for o in ORDINALS
            for a in AMENITY_TYPES
            if f"{o} {a} dist" in df_raw.columns
        ]
        for c in all_amenity_dist:
            df_raw[c] = parse_distance_to_metres(df_raw[c])
        amenity_feat_cols = []
        amenity_parts = []
        for a_type in AMENITY_TYPES:
            a_key = a_type.lower().replace(" ", "_")
            col_1st = f"1st {a_type} dist"
            if col_1st in df_raw.columns:
                feat_name = f"nearest_{a_key}_dist"
                part = df_raw.groupby("ProjectSeries")[col_1st].median().rename(feat_name)
                amenity_parts.append(part)
                amenity_feat_cols.append(feat_name)
            all_ord_cols = [
                f"{o} {a_type} dist" for o in ORDINALS if f"{o} {a_type} dist" in df_raw.columns
            ]
            if all_ord_cols:
                feat_name = f"n_{a_key}_within_{int(AMENITY_DIST_THRESHOLD)}m"
                within_mask = df_raw[all_ord_cols].lt(AMENITY_DIST_THRESHOLD)
                count_col = within_mask.sum(axis=1)
                part = count_col.groupby(df_raw["ProjectSeries"]).median().rename(feat_name)
                amenity_parts.append(part)
                amenity_feat_cols.append(feat_name)
        if amenity_parts:
            amenity_meta = pd.concat(amenity_parts, axis=1).reset_index()
            amenity_meta.columns.name = None
            amenity_meta["project_idx"] = amenity_meta["ProjectSeries"].map(proj_idx_map)
            amenity_meta = amenity_meta.dropna(subset=["project_idx"])
            amenity_meta["project_idx"] = amenity_meta["project_idx"].astype(int)
            series_meta = series_meta.merge(
                amenity_meta[["project_idx"] + amenity_feat_cols], on="project_idx", how="left"
            )
            print(f"After amenity features ({len(amenity_feat_cols)} cols): {series_meta.shape}")
            print(
                f"  FIX #5: count = amenities within {AMENITY_DIST_THRESHOLD}m (not ordinal cols)"
            )
            print(f"  Cols: {amenity_feat_cols}")
            for a_type in AMENITY_TYPES:
                a_key = a_type.lower().replace(" ", "_")
                col = f"nearest_{a_key}_dist"
                if col in series_meta.columns:
                    print(
                        f"  {col:35s} — min={series_meta[col].min():.0f}m  median={series_meta[col].median():.0f}m  max={series_meta[col].max():.0f}m  NaN={series_meta[col].isna().sum()}"
                    )
    else:
        print("No amenity distance columns found — skipping.")
        amenity_feat_cols = []
    print("[sales imputation] source phase 19")
    if has_cols_avail:
        has_meta = (
            df_raw.groupby("ProjectSeries")[has_cols_avail]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0)
            .reset_index()
        )
        has_meta["project_idx"] = has_meta["ProjectSeries"].map(proj_idx_map)
        has_meta = has_meta.dropna(subset=["project_idx"])
        has_meta["project_idx"] = has_meta["project_idx"].astype(int)
        has_meta["facility_count"] = has_meta[has_cols_avail].sum(axis=1)
        series_meta = series_meta.merge(
            has_meta[["project_idx", "facility_count"] + has_cols_avail],
            on="project_idx",
            how="left",
        )
        print(
            f"After Has_* features ({len(has_cols_avail)} cols + facility_count): {series_meta.shape}"
        )
    else:
        print("No Has_* columns found — skipping.")
    print("[sales imputation] source phase 20")
    if school_cols_avail:
        for c in school_cols_avail:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
        school_meta = df_raw.groupby("ProjectSeries")[school_cols_avail].median().reset_index()
        school_meta["project_idx"] = school_meta["ProjectSeries"].map(proj_idx_map)
        school_meta = school_meta.dropna(subset=["project_idx"])
        school_meta["project_idx"] = school_meta["project_idx"].astype(int)
        school_meta["nearest_school_dist"] = school_meta[school_cols_avail].min(axis=1)

        def safe_idxmin(row):
            valid = row.dropna()
            return valid.idxmin() if len(valid) > 0 else np.nan

        school_meta["nearest_school_name"] = school_meta[school_cols_avail].apply(
            safe_idxmin, axis=1
        )
        series_meta = series_meta.merge(
            school_meta[
                ["project_idx", "nearest_school_dist", "nearest_school_name"] + school_cols_avail
            ],
            on="project_idx",
            how="left",
        )
        print(f"After school features ({len(school_cols_avail)} schools): {series_meta.shape}")
    else:
        print("No school columns found — skipping.")
    print("[sales imputation] source phase 21")
    if loc_cols_avail:
        loc_meta = df_raw.groupby("ProjectSeries")[loc_cols_avail].median().reset_index()
        loc_meta["project_idx"] = loc_meta["ProjectSeries"].map(proj_idx_map)
        loc_meta = loc_meta.dropna(subset=["project_idx"])
        loc_meta["project_idx"] = loc_meta["project_idx"].astype(int)
        series_meta = series_meta.merge(
            loc_meta[["project_idx"] + loc_cols_avail], on="project_idx", how="left"
        )
        print(f"After lat/lon: {series_meta.shape}")
    macro_monthly = None
    if macro_cols_avail:
        macro_monthly = df_raw.groupby("YearMonth")[macro_cols_avail].median().reset_index()
        macro_monthly["date"] = macro_monthly["YearMonth"].str[:7]
        macro_monthly = macro_monthly.drop(columns="YearMonth")
        print(f"Macro monthly: {macro_monthly.shape}")
    print(f"\nFinal series_meta: {series_meta.shape[0]} projects × {series_meta.shape[1]} cols")
    print("[sales imputation] source phase 25")
    pre_start_missing = 0
    if settings.top_first:
        from functions.imputation.start_month import apply_top_first

        pre_start_missing = apply_top_first(
            settings.prep_path("property", "property_details_facilities.csv"),
            series_meta, all_months, M_full,
            X_imputed, X_prod_mu2, X_prod_sig2, X_pi_lo, X_pi_hi,
        )
    rows = []
    for p in range(P):
        for t in range(T):
            for b in range(B):
                was_observed = bool(M_full[p, t, b] == 1)
                rows.append(
                    {
                        "project_idx": p,
                        "month_idx": t,
                        "date": dates[t].strftime("%Y-%m"),
                        "sale_psf_obs": float(X_full[p, t, b]) if was_observed else np.nan,
                        "sale_psf_imp": float(X_imputed[p, t, b]),
                        "sale_psf_mu": float(X_prod_mu2[p, t, b]),
                        "sale_psf_pi_lo": float(X_pi_lo[p, t, b]),
                        "sale_psf_pi_hi": float(X_pi_hi[p, t, b]),
                        "decoder_sigma": float(X_prod_sig2[p, t, b]),
                        "was_observed": was_observed,
                    }
                )
    df_long = pd.DataFrame(rows)
    print(f"Long format: {len(df_long):,} rows")
    print("[sales imputation] source phase 26")
    comp_map = series_meta.set_index("project_idx")["Condo_Age_2026"].apply(
        lambda a: 2026 - a if pd.notna(a) else np.nan
    )
    df_long["condo_age_at_transaction"] = (
        pd.to_datetime(df_long["date"]).dt.year - df_long["project_idx"].map(comp_map)
    ).clip(lower=0)
    print(
        f"condo_age_at_transaction NaN: {df_long['condo_age_at_transaction'].isna().sum()} ({100 * df_long['condo_age_at_transaction'].isna().mean():.1f}%)"
    )
    print("[sales imputation] source phase 27")
    amenity_cols_in_meta = [
        c
        for c in series_meta.columns
        if c.startswith("nearest_")
        and "mrt" not in c.lower()
        or (c.startswith("n_") and "within" in c)
    ]
    school_cols_in_meta = ["nearest_school_dist", "nearest_school_name"] + [
        c for c in school_cols_avail if c in series_meta.columns
    ]
    has_cols_in_meta = ["facility_count"] + [c for c in has_cols_avail if c in series_meta.columns]
    loc_cols_in_meta = [c for c in loc_cols_avail if c in series_meta.columns]
    meta_join = (
        [
            "project_idx",
            "condo_name",
            "size_tier",
            "size_label",
            "district",
            "segment",
            "area_sqft_med",
            "n_transactions",
        ]
        + [c for c in prop_cols_avail if c in series_meta.columns]
        + [c for c in ["nearest_mrt_dist", "nearest_mrt_name"] if c in series_meta.columns]
        + [c for c in amenity_cols_in_meta if c in series_meta.columns]
        + [c for c in school_cols_in_meta if c in series_meta.columns]
        + [c for c in has_cols_in_meta if c in series_meta.columns]
        + loc_cols_in_meta
    )
    meta_join = list(dict.fromkeys(meta_join))
    df_full_out = df_long.merge(series_meta[meta_join], on="project_idx", how="left")
    if macro_monthly is not None:
        df_full_out = df_full_out.merge(macro_monthly, on="date", how="left")
    print(f"Final DataFrame: {len(df_full_out):,} rows × {len(df_full_out.columns)} columns")
    print(f"Missing sale_psf_imp : {df_full_out['sale_psf_imp'].isna().sum()}")
    print(f"Date range           : {df_full_out['date'].min()} → {df_full_out['date'].max()}")
    print("[sales imputation] source phase 29")
    actual_districts = set(
        pd.to_numeric(df_full_out["district"], errors="coerce").dropna().astype(int).unique()
    )
    duplicate_panel_keys = int(df_full_out.duplicated(["project_idx", "date"]).sum())
    missing_sale_imp = int(df_full_out["sale_psf_imp"].isna().sum())
    print(f"Output districts     : {sorted(actual_districts)}")
    print(f"Missing districts    : {sorted(EXPECTED_DISTRICTS - actual_districts)}")
    print(f"Unexpected districts : {sorted(actual_districts - EXPECTED_DISTRICTS)}")
    print(f"Duplicate panel keys : {duplicate_panel_keys:,}")
    print(f"Missing sale_psf_imp : {missing_sale_imp:,}")
    print(
        f"Sale PSF imputed range: [{df_full_out['sale_psf_imp'].min():.2f}, {df_full_out['sale_psf_imp'].max():.2f}]"
    )
    assert actual_districts == EXPECTED_DISTRICTS
    assert duplicate_panel_keys == 0
    assert missing_sale_imp == pre_start_missing
    df_full_out.to_csv(CSV_OUT, index=False)
    print(f"Saved {CSV_OUT}  ({os.path.getsize(CSV_OUT) / 1000000.0:.1f} MB)")
    bundle_out = {
        "schema_version": 4,
        "producer_version": "vae_a4_cs_mcra_v6",
        "model_config": best_config,
        "q_hat": float(q_hat_prod),
        "P": int(P),
        "T": int(T),
        "B": int(B),
        "X_full": np.asarray(X_full, dtype=np.float32),
        "M_full": np.asarray(M_full, dtype=np.uint8),
        "X_imputed": np.asarray(X_imputed, dtype=np.float32),
        "X_eval_imputed": np.asarray(X_eval_imputed, dtype=np.float32),
        "X_pi_lo": np.asarray(X_pi_lo, dtype=np.float32),
        "X_pi_hi": np.asarray(X_pi_hi, dtype=np.float32),
        "X_decoder_mu": np.asarray(X_prod_mu2, dtype=np.float32),
        "X_decoder_sigma": np.asarray(X_prod_sig2, dtype=np.float32),
        "series_meta": series_meta,
        "all_projects": all_projects,
        "all_months": all_months,
        "macro_monthly": macro_monthly,
        "val_es_pos": val_es_pos,
        "cal_pos": cal_pos,
        "test_pos": test_pos,
        "train_pos": train_pos,
    }
    save_artifacts(
        settings, "sales", bundle_out, results=results, coverage=coverage,
        eval_vms=eval_vms, prod_vms=prod_vms, context_matrix=C_ctx,
        psf_reasonable_range=PSF_REASONABLE_RANGE, started_at=started_at,
    )
    return CSV_OUT
