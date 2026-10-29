"""Optional raw rebuild migrated from data_prep/sale_data/1_merging_data_sale.ipynb. See README for missing original reference requirements."""


def run(settings):
    from functions.config import PrepPaths

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
    print("[raw preprocessing] source phase 2")
    import os
    import pandas as pd

    combined_df = pd.read_csv(paths.file("raw_realis_transaction.csv"))
    total_rows = len(combined_df)
    combined_df = (
        combined_df[combined_df["Property Type"].isin(["Condominium", "Apartment"])]
        .copy()
        .reset_index(drop=True)
    )
    print(f"Filtered rows: {len(combined_df):,}")
    unique_districts = combined_df["Postal District"].unique()
    num_districts = len(unique_districts)
    combined_df["Sale_Date"] = pd.to_datetime(combined_df["Sale Date"], errors="coerce")
    for col in [
        "Transacted Price ($)",
        "Nett Price($)",
        "Unit Price ($ PSF)",
        "Unit Price ($ PSM)",
    ]:
        combined_df[col] = pd.to_numeric(
            combined_df[col].astype(str).str.replace(",", ""), errors="coerce"
        )
    district_analysis = (
        combined_df.groupby("Postal District")
        .agg(
            Total_Records=("Project Name", "count"),
            Start_Date=("Sale_Date", "min"),
            End_Date=("Sale_Date", "max"),
            Avg_Transacted_Price=("Transacted Price ($)", "mean"),
            Avg_Unit_Price_PSF=("Unit Price ($ PSF)", "mean"),
        )
        .reset_index()
    )

    def calculate_monthly_avg(row):
        if pd.isnull(row["Start_Date"]) or pd.isnull(row["End_Date"]):
            return 0
        months = (row["End_Date"].year - row["Start_Date"].year) * 12 + (
            row["End_Date"].month - row["Start_Date"].month
        )
        return round(row["Total_Records"] / max(months, 1), 2)

    district_analysis["Avg_Records_Per_Month"] = district_analysis.apply(
        calculate_monthly_avg, axis=1
    )
    type_of_sale_counts = combined_df["Type of Sale"].value_counts()
    property_type_counts = combined_df["Property Type"].value_counts()
    print("=== URA SALE DATA REPORT ===")
    print(f"1. No. rows: {total_rows:,}")
    print(f"2. Filtered rows: {len(combined_df):,}")
    print(f"3. No. Districts: {num_districts} ({', '.join(map(str, sorted(unique_districts)))})")
    print(f"\n4. Type of Sale:\n{type_of_sale_counts}")
    print(f"\n5. Property Type:\n{property_type_counts}")
    print("\n6. Analysis by District:")
    print(
        district_analysis[
            [
                "Postal District",
                "Start_Date",
                "End_Date",
                "Total_Records",
                "Avg_Records_Per_Month",
                "Avg_Transacted_Price",
            ]
        ]
    )
    print("[raw preprocessing] source phase 4")
    print("All columns:", combined_df.columns.tolist())
    if "district" in combined_df.columns:
        combined_df = combined_df.loc[:, :"district"]
    else:
        print("WARNING: 'district' column not found — skipping trim step")
    print("\nKept columns:", combined_df.columns.tolist())
    print("[raw preprocessing] source phase 5")
    import pandas as pd

    lookup_file = paths.file("../URA_full/condo_latlon_onemap.csv")
    fallback_file = paths.file("sale_missing_project_mrt_candidates_walk.csv")
    output_name = paths.file("URA_sale_address.csv")
    missing_file = paths.file("sale_missing_projects.csv")
    lookup_df = pd.read_csv(lookup_file)
    lookup_df.columns = lookup_df.columns.str.strip()
    lookup_df = lookup_df[["project_name", "onemap_address", "latitude", "longitude"]]
    lookup_df = lookup_df.drop_duplicates(subset=["project_name"])
    fallback_df = pd.read_csv(fallback_file)
    fallback_df.columns = fallback_df.columns.str.strip()
    fallback_lookup = (
        fallback_df[["project_name", "condo_lat", "condo_lng"]]
        .drop_duplicates(subset=["project_name"])
        .rename(columns={"condo_lat": "latitude", "condo_lng": "longitude"})
    )
    fallback_lookup["onemap_address"] = None
    combined_lookup = pd.concat([lookup_df, fallback_lookup], ignore_index=True)
    combined_lookup = combined_lookup.drop_duplicates(subset=["project_name"], keep="first")
    print(f"Primary lookup  : {len(lookup_df):,} projects")
    print(f"Fallback lookup : {len(fallback_lookup):,} projects")
    print(f"Combined lookup : {len(combined_lookup):,} projects")
    print(f"\nMapping coordinates for {len(combined_df):,} rows...")
    final_df = pd.merge(
        combined_df, combined_lookup, left_on="Project Name", right_on="project_name", how="left"
    )
    final_df = final_df.drop(columns=["project_name"])
    unmatched_mask = final_df["latitude"].isna()
    missing_rows_count = unmatched_mask.sum()
    if missing_rows_count > 0:
        missing_projects = final_df[unmatched_mask][
            ["Project Name", "Postal District"]
        ].drop_duplicates()
        missing_projects.to_csv(missing_file, index=False, encoding="utf-8-sig")
        print(f"WARNING: {missing_rows_count:,} rows still missing coordinates.")
        print(f"Missing project list saved to: {missing_file}")
    else:
        print("All projects matched successfully!")
    final_df = final_df.dropna(subset=["latitude"])
    print(f"Rows after removing unmatched: {len(final_df):,}")
    cols = final_df.columns.tolist()
    if "Project Name" in cols:
        idx = cols.index("Project Name") + 1
        new_info = ["onemap_address", "latitude", "longitude"]
        other_cols = [c for c in cols if c not in new_info]
        final_column_order = other_cols[:idx] + new_info + other_cols[idx:]
        final_df = final_df[final_column_order]
    final_df.to_csv(output_name, index=False, encoding="utf-8-sig")
    print(f"SUCCESS: Saved to {output_name}")
    print("[raw preprocessing] source phase 6")
    import pandas as pd
    import numpy as np

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

    input_file = paths.file("URA_sale_address.csv")
    school_file = paths.file("top30_primary_schools.csv")
    output_file = paths.file("URA_sale_primary_schools.csv")
    df_schools = pd.read_csv(school_file)
    df_addr = pd.read_csv(input_file)
    df_addr = df_addr.dropna(subset=["latitude", "longitude"]).copy()
    print(f"🔄 Calculating distances for {len(df_addr):,} rows...")
    for _, school in df_schools.iterrows():
        name = school["School Name"]
        dist = haversine_vectorized(
            df_addr["latitude"].values,
            df_addr["longitude"].values,
            school["Latitude"],
            school["Longitude"],
        )
        df_addr[name] = np.where(dist <= 2.0, dist.astype(np.float32), np.nan)
    df_addr.to_csv(output_file, index=False)
    print(f"✅ Done! Saved to: {output_file}")
    print("[raw preprocessing] source phase 7")
    import pandas as pd

    df_mrt_full = pd.read_csv(paths.file("full_period_condo_mrt_candidates_walk.csv"))
    df_mrt_missing = pd.read_csv(paths.file("sale_missing_project_mrt_candidates_walk.csv"))
    projects_in_full = set(df_mrt_full["project_name"].unique())
    df_mrt_missing = df_mrt_missing[
        ~df_mrt_missing["project_name"].isin(projects_in_full)
        & df_mrt_missing["candidate_name"].notna()
    ]
    print(f"Full file rows        : {len(df_mrt_full):,}")
    print(f"Fallback rows (new)   : {len(df_mrt_missing):,}")
    print(f"New projects added    : {df_mrt_missing['project_name'].nunique():,}")
    df_mrt = pd.concat([df_mrt_full, df_mrt_missing], ignore_index=True)
    print(f"Combined rows         : {len(df_mrt):,}")
    unique_stations = df_mrt["candidate_name"].unique()
    print(f"\nNo. MRT stations: {len(unique_stations)}")
    print(df_mrt["candidate_name"].value_counts().head(10))
    print("[raw preprocessing] source phase 8")
    import pandas as pd

    input_file = paths.file("URA_sale_primary_schools.csv")
    output_file = paths.file("URA_sale_schools_mrt.csv")
    print("🔄 Processing MRT data...")
    mrt_pivot = df_mrt.pivot_table(
        index="project_name", columns="candidate_name", values="walk_distance_m"
    ).reset_index()
    mrt_pivot = mrt_pivot.add_suffix("_MRT")
    mrt_pivot = mrt_pivot.rename(columns={"project_name_MRT": "project_name"})
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
    else:
        print(f"🔄 Reading {input_file}...")
        df_ura = pd.read_csv(input_file)
        print("🔗 Merging MRT distances...")
        combined_df = pd.merge(
            df_ura, mrt_pivot, left_on="Project Name", right_on="project_name", how="left"
        )
        if "project_name" in combined_df.columns:
            combined_df = combined_df.drop(columns=["project_name"])
        combined_df.to_csv(output_file, index=False)
        print("-" * 40)
        print("✅ PROCESS COMPLETE")
        print(f"Final File : {output_file}")
        print(f"Total Rows : {len(combined_df):,}")
        print("-" * 40)
    print("[raw preprocessing] source phase 12")
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    print("📂 Loading datasets...")
    ura_df = pd.read_csv(paths.file("URA_sale_schools_mrt.csv"))
    macro_df = pd.read_csv(paths.file("rental_merged_tenure_1999-2026_enriched_macro.csv"))
    sg_macro_df = pd.read_csv(paths.file("sg_macro_monthly.csv"))
    print("🔍 Filtering project macro columns...")
    cols = macro_df.columns.tolist()
    start_idx = cols.index("Lease Commencement Date") + 1
    selected_macro_cols = ["Project Name"] + cols[start_idx:]
    macro_filtered = macro_df[selected_macro_cols].drop_duplicates(subset=["Project Name"])
    print("📅 Parsing Sale Date...")
    ura_df["Sale_Date"] = pd.to_datetime(ura_df["Sale Date"], errors="coerce")
    sg_macro_df["month_dt"] = pd.to_datetime(sg_macro_df["month"], errors="coerce")
    ura_df["merge_key"] = ura_df["Sale_Date"].dt.to_period("M")
    sg_macro_df["merge_key"] = sg_macro_df["month_dt"].dt.to_period("M")
    sg_macro_filtered = sg_macro_df.drop(columns=["month", "month_dt"])
    print("\n🔄 Merging datasets...")
    with tqdm(total=2, desc="Merge Progress") as pbar:
        df_step1 = pd.merge(ura_df, macro_filtered, on="Project Name", how="left")
        pbar.update(1)
        final_df = pd.merge(df_step1, sg_macro_filtered, on="merge_key", how="left")
        pbar.update(1)
    final_df = final_df.drop(columns=["merge_key"])
    output_name = paths.file("URA_sale_final_enriched.csv")
    print(f"\n💾 Saving to {output_name}...")
    chunk_size = 5000
    num_chunks = (len(final_df) + chunk_size - 1) // chunk_size
    with open(output_name, "w", encoding="utf-8", newline="") as f:
        for i in tqdm(range(num_chunks), desc="Writing CSV"):
            start = i * chunk_size
            end = min(start + chunk_size, len(final_df))
            chunk = final_df.iloc[start:end]
            chunk.to_csv(f, index=False, header=i == 0)
    print("-" * 40)
    print("✅ Process Finished!")
    print(f"Total rows        : {len(final_df):,}")
    print(f"Invalid Sale Dates: {ura_df['Sale_Date'].isna().sum():,}")
    print("-" * 40)
    print("[raw preprocessing] source phase 14")
    import re
    import numpy as np
    import pandas as pd

    REFERENCE_YEAR = 2026

    def normalize_project_name(x):
        if x is None:
            return None
        s = str(x).strip().upper()
        if not s:
            return None
        s = re.sub("[^A-Z0-9 ]+", " ", s)
        s = re.sub("\\s+", " ", s).strip()
        return s or None

    def extract_year(x):
        if x is None:
            return None
        m = re.search("(19\\d{2}|20\\d{2})", str(x))
        return int(m.group(1)) if m else None

    def parse_project_size(x):
        if pd.isna(x):
            return (None, None, None)
        s = str(x).strip().lower()
        if "small" in s:
            return (1.0, 0.0, 0.0)
        elif "medium" in s:
            return (0.0, 1.0, 0.0)
        elif "large" in s:
            return (0.0, 0.0, 1.0)
        return (None, None, None)

    print("Helper functions defined.")
    print("[raw preprocessing] source phase 15")
    prop = pd.read_csv(paths.file("./property_details_facilities.csv"), low_memory=False)
    print(f"99.co raw shape: {prop.shape}")
    prop_project_col = "Project Name"
    prop_tenure_col = "Tenure"
    prop_top_col = "TOP date"
    prop_size_col = "Project Size"
    prop_neigh_col = "Neighbourhood"
    facility_start = list(prop.columns).index("Adventure Park")
    raw_facility_cols = prop.columns[facility_start:].tolist()
    print(f"Facility columns found: {len(raw_facility_cols)}")
    rename_map = {c: f"Has_{c}" for c in raw_facility_cols}
    prop = prop.rename(columns=rename_map)
    facility_cols = [f"Has_{c}" for c in raw_facility_cols]
    print(f"Renamed to Has_*: {facility_cols[:5]} ...")
    print("[raw preprocessing] source phase 16")
    keep_cols = [
        prop_project_col,
        prop_tenure_col,
        prop_top_col,
        prop_size_col,
        prop_neigh_col,
    ] + facility_cols
    keep_cols = [c for c in keep_cols if c in prop.columns]
    proj_tbl = prop[keep_cols].copy()
    proj_tbl["_proj_key"] = proj_tbl[prop_project_col].map(normalize_project_name)
    proj_tbl = proj_tbl.dropna(subset=["_proj_key"])
    proj_tbl = proj_tbl.drop_duplicates(subset=["_proj_key"], keep="first")
    proj_tbl["_top_year"] = (
        proj_tbl[prop_top_col].map(extract_year) if prop_top_col in proj_tbl.columns else np.nan
    )
    proj_tbl["house_age_2026"] = proj_tbl["_top_year"].apply(
        lambda y: REFERENCE_YEAR - int(y) if pd.notna(y) else np.nan
    )
    size_parsed = (
        proj_tbl[prop_size_col].apply(parse_project_size)
        if prop_size_col in proj_tbl.columns
        else [(None, None, None)] * len(proj_tbl)
    )
    proj_tbl["project_size_small"] = [x[0] for x in size_parsed]
    proj_tbl["project_size_medium"] = [x[1] for x in size_parsed]
    proj_tbl["project_size_large"] = [x[2] for x in size_parsed]
    if prop_neigh_col in proj_tbl.columns:
        proj_tbl["Neighbourhood"] = proj_tbl[prop_neigh_col].astype(str).str.strip()
        proj_tbl["Neighbourhood"] = proj_tbl["Neighbourhood"].replace("nan", np.nan)
    print(f"Project lookup table: {proj_tbl.shape}")
    print("[raw preprocessing] source phase 17")
    print("Loading URA_sale_final_enriched.csv ...")
    df = pd.read_csv(paths.file("./URA_sale_final_enriched.csv"), low_memory=False).copy()
    print(f"Sale data shape: {df.shape}")
    df["_proj_key"] = df["Project Name"].map(normalize_project_name)
    new_cols = [
        "_proj_key",
        "house_age_2026",
        "project_size_small",
        "project_size_medium",
        "project_size_large",
        "Neighbourhood",
    ] + facility_cols
    new_cols = [c for c in new_cols if c in proj_tbl.columns]
    df_final = df.merge(proj_tbl[new_cols], on="_proj_key", how="left")
    df_final = df_final.drop(columns=["_proj_key"])
    print(f"Shape after merge : {df_final.shape}")
    print(f"Row count unchanged: {len(df_final) == len(df)}")
    matched = df_final["Has_Gym"].notna().sum() if "Has_Gym" in df_final.columns else 0
    print(f"Amenity match rate: {matched}/{len(df_final)} ({matched / len(df_final) * 100:.1f}%)")
    print("[raw preprocessing] source phase 18")
    output_path = paths.file("./URA_sale_final_enriched_v2.csv")
    df_final.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"Final shape: {df_final.shape}")
