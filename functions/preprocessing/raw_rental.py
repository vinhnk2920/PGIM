"""Optional raw rebuild migrated from data_prep/data_v1/data_pipeline_complete_v3.ipynb. See README for missing original reference requirements."""


def run(settings):
    from functions.config import PrepPaths
    from functions.preprocessing.merging import merge_reference

    paths = PrepPaths(
        settings,
        generated={
            "URA_sale_data.csv",
            "URA_sale_schools_mrt.csv",
            "URA_sale_final_enriched.csv",
            "sale_missing_projects.csv",
            "URA_sale_final_enriched_v2.csv",
            "URA_sale_address.csv",
            "URA_merged_full_v3_new.csv",
            "URA_sale_primary_schools.csv",
        },
    )
    print("[raw preprocessing] source phase 1")
    import pandas as pd

    df_check = pd.read_csv(paths.file("URA_merged_full_v1.csv"), nrows=0)
    print("[raw preprocessing] source phase 2")
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    print("[STEP 1] Loading raw URA rental data...")
    raw_df = pd.read_csv(paths.file("rental_merged_tenure_1999-2026.csv"))
    print(f"  Loaded: {len(raw_df):,} rows, {len(raw_df.columns)} columns")
    raw_df["Lease Commencement Date"] = raw_df["Lease Commencement Date"].str.replace(
        "Sept", "Sep", regex=False
    )
    raw_df["Lease Commencement Date"] = pd.to_datetime(
        raw_df["Lease Commencement Date"], format="%b-%y", errors="coerce"
    )
    nat_count = raw_df["Lease Commencement Date"].isna().sum()
    print(f"  Date parsing: {nat_count} NaT errors")
    print(f"  Ready: {len(raw_df):,} rows (before filtering)")
    print("[raw preprocessing] source phase 4")
    print("[STEP 2] Adding 99.co condo enrichments...")
    df_99co = pd.read_csv(paths.file("URA_enriched_with_99co_v2.csv"), low_memory=False)
    df_99co["Project Name"] = df_99co["Project Name"].astype(str).str.strip().str.upper()
    raw_df["Project Name"] = raw_df["Project Name"].astype(str).str.strip().str.upper()
    if "Lease Commencement Date" in df_99co.columns:
        df_99co["Lease Commencement Date"] = pd.to_datetime(
            df_99co["Lease Commencement Date"], errors="coerce"
        )
        raw_df["Lease Commencement Date"] = pd.to_datetime(
            raw_df["Lease Commencement Date"], errors="coerce"
        )
    additional_cols = [
        col
        for col in df_99co.columns
        if col not in raw_df.columns and col not in ["Project Name", "Lease Commencement Date"]
    ]
    if "Lease Commencement Date" in df_99co.columns and "Lease Commencement Date" in raw_df.columns:
        merge_keys = ["Project Name", "Lease Commencement Date"]
    else:
        merge_keys = ["Project Name"]
    print(f"  99.co merge keys: {merge_keys}")
    print(f"  99.co additional columns: {len(additional_cols):,}")
    raw_df = pd.merge(
        raw_df,
        df_99co[merge_keys + additional_cols].drop_duplicates(subset=merge_keys),
        on=merge_keys,
        how="left",
    )
    print(f"  Rows after 99.co merge: {len(raw_df):,}")
    print(f"  Columns after 99.co merge: {len(raw_df.columns):,}")
    print("[raw preprocessing] source phase 5")
    print("\n[STEP 3] Mapping addresses & coordinates...")
    lookup_df = pd.read_csv(paths.file("condo_latlon_onemap.csv"))
    lookup_df.columns = lookup_df.columns.str.strip()
    lookup_cols = ["project_name", "onemap_address", "latitude", "longitude"]
    lookup_df = lookup_df[lookup_cols].drop_duplicates(subset=["project_name"])
    df_with_addr = merge_reference(
        raw_df, lookup_df, left_on="Project Name", right_on="project_name"
    )
    missing_addr_df = df_with_addr[df_with_addr["latitude"].isna()]
    num_missing_condos = missing_addr_df["Project Name"].nunique()
    print(f"  --> Total unique missing condos: {num_missing_condos:,}")
    if "project_name" in df_with_addr.columns:
        df_with_addr = df_with_addr.drop(columns=["project_name"])
    missing_count = df_with_addr["latitude"].isna().sum()
    df_with_addr = df_with_addr.dropna(subset=["latitude"])
    print(f"\n  Missing coordinates: {missing_count:,} rows (removed)")
    print(f"  Rows after address merge: {len(df_with_addr):,}")
    print("[raw preprocessing] source phase 6")
    print("\n[STEP 4] Calculating school distances (5km radius)...")

    def haversine_vectorized(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = (
            np.radians(lat1),
            np.radians(lon1),
            np.radians(lat2),
            np.radians(lon2),
        )
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(a))

    schools_df = pd.read_csv(paths.file("top30_primary_schools.csv"))
    print(f"  Schools to process: {len(schools_df)}")
    for _, school in tqdm(schools_df.iterrows(), total=len(schools_df), desc="Schools"):
        name = school["School Name"]
        dist = haversine_vectorized(
            df_with_addr["latitude"].values,
            df_with_addr["longitude"].values,
            school["Latitude"],
            school["Longitude"],
        )
        df_with_addr[name] = np.where(dist <= 2.0, dist.astype(np.float32), np.nan)
    school_cols = schools_df["School Name"].tolist()
    no_school_mask = df_with_addr[school_cols].isna().all(axis=1)
    no_school_count = int(no_school_mask.sum())
    no_school_projects = df_with_addr.loc[no_school_mask, "Project Name"].nunique()
    print(
        f"  Condos with no nearby school within 2km: {no_school_count:,} rows, {no_school_projects:,} unique projects"
    )
    if no_school_count > 0:
        print("  Sample projects without any nearby school:")
        print(df_with_addr.loc[no_school_mask, "Project Name"].drop_duplicates().head(20).tolist())
    print(f"  Done. Rows: {len(df_with_addr):,}")
    print("[raw preprocessing] source phase 7")
    print("\n[STEP 5] Merging MRT distances...")
    mrt_df = pd.read_csv(paths.file("full_period_condo_mrt_candidates_walk.csv"))
    mrt_pivot = mrt_df.pivot_table(
        index="project_name", columns="candidate_name", values="walk_distance_m"
    ).reset_index()
    mrt_pivot = mrt_pivot.add_suffix("_MRT")
    mrt_pivot = mrt_pivot.rename(columns={"project_name_MRT": "project_name"})
    df_with_schools_mrt = merge_reference(
        df_with_addr, mrt_pivot, left_on="Project Name", right_on="project_name"
    )
    if "project_name" in df_with_schools_mrt.columns:
        df_with_schools_mrt = df_with_schools_mrt.drop(columns=["project_name"])
    mrt_cols = [col for col in df_with_schools_mrt.columns if col.endswith("_MRT")]
    no_mrt_mask = df_with_schools_mrt[mrt_cols].isna().all(axis=1)
    no_mrt_count = int(no_mrt_mask.sum())
    no_mrt_projects = df_with_schools_mrt.loc[no_mrt_mask, "Project Name"].nunique()
    print(
        f"  Condos with no nearby MRT candidate: {no_mrt_count:,} rows, {no_mrt_projects:,} unique projects"
    )
    if no_mrt_count > 0:
        print("  Sample projects without any nearby MRT:")
        print(
            df_with_schools_mrt.loc[no_mrt_mask, "Project Name"].drop_duplicates().head(20).tolist()
        )
    print(f"  MRT columns added: {len(mrt_pivot.columns) - 1}")
    print(f"  Rows: {len(df_with_schools_mrt):,}")
    print("[raw preprocessing] source phase 8")
    print("\n[STEP 6] Adding project-level macro data...")
    macro_df = pd.read_csv(paths.file("rental_merged_tenure_1999-2026_enriched_macro.csv"))
    df_with_schools_mrt["Project Name"] = (
        df_with_schools_mrt["Project Name"].astype(str).str.strip().str.upper()
    )
    macro_df["Project Name"] = macro_df["Project Name"].astype(str).str.strip().str.upper()
    overlapping_cols = [
        c for c in macro_df.columns if c in df_with_schools_mrt.columns and c != "Project Name"
    ]
    if overlapping_cols:
        macro_df = macro_df.drop(columns=overlapping_cols)
    cols = macro_df.columns.tolist()
    if "Lease Commencement Date" in cols:
        start_idx = cols.index("Lease Commencement Date") + 1
        selected_macro_cols = ["Project Name"] + cols[start_idx:]
    else:
        selected_macro_cols = macro_df.columns.tolist()
    macro_filtered = macro_df[selected_macro_cols].drop_duplicates(subset=["Project Name"])
    df_with_project_macro = pd.merge(
        df_with_schools_mrt, macro_filtered, on="Project Name", how="left"
    )
    print(f"  Macro columns added: {len(macro_filtered.columns) - 1}")
    print(f"  Rows: {len(df_with_project_macro):,}")
    macro_cols = [col for col in macro_filtered.columns if col != "Project Name"]
    missing_macro = df_with_project_macro[macro_cols].isnull().sum().sort_values(ascending=False)
    print("\n  Missing values in project-level macro columns:")
    print(missing_macro[missing_macro > 0].head(20))
    print(f"  Total columns with missing values: {(missing_macro > 0).sum()}")
    print("\n[TRACKING] Detailed missing data analysis...")
    if "Condo_Age_2026" in df_with_project_macro.columns:
        missing_projects_df = df_with_project_macro[
            df_with_project_macro["Condo_Age_2026"].isnull()
        ]
        num_missing_projects = missing_projects_df["Project Name"].nunique()
        total_projects = df_with_project_macro["Project Name"].nunique()
        print(
            f"  Projects missing 'Condo_Age_2026': {num_missing_projects} / {total_projects} projects ({num_missing_projects / total_projects * 100:.1f}%)"
        )
        sample_missing_projects = missing_projects_df["Project Name"].dropna().unique()[:10]
        print(f"  Sample of 10 missing projects: {list(sample_missing_projects)}")
    print("\n[TRACKING] Detailed missing data analysis for selected columns...")
    columns_to_check = [
        "sora_compounded_3m_pct",
        "sora_overnight_pct",
        "ura_private_rental_index_yoy_pct",
        "ura_private_price_index_avg_3seg_yoy_pct",
        "Large_Dev_200plus",
        "tenure_remaining_years",
    ]
    time_col = "Lease Commencement Date"
    for col in columns_to_check:
        print(f"\n--- Checking Column: {col} ---")
        if col not in df_with_project_macro.columns:
            print(f"  [!] Column '{col}' not found in the dataset.")
            continue
        missing_df = df_with_project_macro[df_with_project_macro[col].isnull()]
        missing_count = len(missing_df)
        total_count = len(df_with_project_macro)
        if missing_count == 0:
            print(f"  Excellent! No missing data for '{col}'.")
            continue
        print(
            f"  Missing rows: {missing_count:,} / {total_count:,} ({missing_count / total_count * 100:.1f}%)"
        )
        if col in ["Large_Dev_200plus", "tenure_remaining_years", "Condo_Age_2026"]:
            num_missing_projects = missing_df["Project Name"].nunique()
            print(f"  Missing projects: {num_missing_projects:,}")
            sample_missing_projects = missing_df["Project Name"].dropna().unique()[:5]
            print(f"  Sample missing projects: {list(sample_missing_projects)}")
        elif time_col in df_with_project_macro.columns:
            missing_by_time = missing_df.groupby(time_col).size().sort_values(ascending=False)
            print("  Top 5 time periods with the most missing data:")
            for time_period, count in missing_by_time.head(5).items():
                print(f"    - {time_period}: missing {count:,} rows")
            missing_dates = pd.to_datetime(missing_df[time_col], errors="coerce").dropna()
            if not missing_dates.empty:
                print(
                    f"  Missing data spans from: {missing_dates.min().strftime('%Y-%m')} to {missing_dates.max().strftime('%Y-%m')}"
                )
    print("[raw preprocessing] source phase 9")
    print("\n[STEP 7] Adding monthly macro data...")
    sg_macro_df = pd.read_csv(paths.file("sg_macro_monthly.csv"))
    sg_macro_df["month_dt"] = pd.to_datetime(sg_macro_df["month"], errors="coerce")
    sg_macro_df["merge_key"] = sg_macro_df["month_dt"].dt.to_period("M")
    sg_macro_filtered = sg_macro_df.drop(columns=["month", "month_dt"])
    df_with_project_macro["merge_key"] = df_with_project_macro[
        "Lease Commencement Date"
    ].dt.to_period("M")
    df_with_all_macro = pd.merge(
        df_with_project_macro,
        sg_macro_filtered,
        on="merge_key",
        how="left",
        suffixes=("", "_monthly"),
    )
    df_with_all_macro = df_with_all_macro.drop(columns=["merge_key"])
    monthly_macro_columns = []
    for col in sg_macro_filtered.columns:
        if col == "merge_key":
            continue
        if col in df_with_all_macro.columns:
            monthly_macro_columns.append(col)
        elif f"{col}_monthly" in df_with_all_macro.columns:
            monthly_macro_columns.append(f"{col}_monthly")
        else:
            print(f"  WARNING: monthly macro column missing after merge: {col}")
    print(f"  Monthly macro columns added: {len(monthly_macro_columns)}")
    print(f"  Rows: {len(df_with_all_macro):,}")
    print("\n  Missing values in monthly macro columns:")
    monthly_macro_missing = (
        df_with_all_macro[monthly_macro_columns].isnull().sum().sort_values(ascending=False)
    )
    print(monthly_macro_missing[monthly_macro_missing > 0].head(20))
    print("[raw preprocessing] source phase 10")
    print("\n[STEP 8] Adding amenity data...")

    def clean_name(name):
        if pd.isna(name):
            return name
        return str(name).strip().upper().replace("#", "").replace("  ", " ")

    amenities_df = pd.read_csv(paths.file("nearby_5km_amenities.csv"))
    columns_to_exclude = ["MRT", "Schools", "URL"]
    cols_to_keep = [
        col
        for col in amenities_df.columns
        if not any((word in col for word in columns_to_exclude))
        and (not any((x in col for x in ["6th", "7th", "8th", "9th", "10th"])))
        and (col != "Project Name")
    ]
    amenities_filtered = amenities_df[cols_to_keep].copy()
    amenities_filtered["Project_Key"] = amenities_filtered["Project Name in Realis"].apply(
        clean_name
    )
    df_with_all_macro["Project_Key"] = df_with_all_macro["Project Name"].apply(clean_name)
    df_final = merge_reference(df_with_all_macro, amenities_filtered, on="Project_Key")
    df_final = df_final.drop(columns=["Project_Key"])
    amenity_cols = [col for col in amenities_filtered.columns if col != "Project_Key"]
    missing_amenities = df_final[amenity_cols].isnull().sum().sort_values(ascending=False)
    rows_without_amenities = df_final[amenity_cols].isnull().all(axis=1).sum()
    print(f"  Amenity columns added: {len(amenities_filtered.columns) - 1}")
    print(f"  Rows: {len(df_final):,}")
    print("\n  Missing values in amenity columns:")
    print(missing_amenities[missing_amenities > 0])
    print(f"\n  Rows with no amenity values at all: {rows_without_amenities:,}")
    print("[raw preprocessing] source phase 11")
    print("\n[STEP 9] Comparing final columns to URA_merged_full_v1.csv...")
    v1_columns = df_check.columns.tolist()
    final_columns = df_final.columns.tolist()
    missing_in_final = [c for c in v1_columns if c not in final_columns]
    extra_in_final = [c for c in final_columns if c not in v1_columns]
    common_columns = [c for c in v1_columns if c in final_columns]
    print(f"  Columns in URA_merged_full_v1.csv: {len(v1_columns):,}")
    print(f"  Columns in current final dataset: {len(final_columns):,}")
    print(f"  Missing compared to v1: {len(missing_in_final):,}")
    print(f"  Extra columns not in v1: {len(extra_in_final):,}")
    if missing_in_final:
        print("  Missing columns from final dataset:")
        print(missing_in_final)
    if extra_in_final:
        print("  Extra columns in final dataset:")
        print(extra_in_final[:50])
        if len(extra_in_final) > 50:
            print(f"  ... and {len(extra_in_final) - 50} more")
    df_final = df_final[[*common_columns, *extra_in_final]]
    print(f"  Final dataframe columns reordered. New total columns: {len(df_final.columns):,}")
    print("[raw preprocessing] source phase 12")
    print("\n[STEP 10] Saving final dataset...")
    output_file = paths.file("URA_merged_full_v3.csv")
    chunk_size = 5000
    num_chunks = (len(df_final) + chunk_size - 1) // chunk_size
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        for i in tqdm(range(num_chunks), desc="Writing CSV"):
            start = i * chunk_size
            end = min(start + chunk_size, len(df_final))
            chunk = df_final.iloc[start:end]
            chunk.to_csv(f, index=False, header=i == 0)
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print(f"Total rows: {len(df_final):,}")
    print(f"Total columns: {len(df_final.columns)}")
    print(
        f"Date range: {df_final['Lease Commencement Date'].min()} to {df_final['Lease Commencement Date'].max()}"
    )
    print(f"Districts: {df_final['Postal District'].nunique()}")
    print("=" * 60)
