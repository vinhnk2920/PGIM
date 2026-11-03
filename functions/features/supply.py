"""supply. Explicit callable; no work is performed on import."""

from functions.common import display
from functions.features.supply_data import load_supply_table
from functions.preprocessing.monthly import fill_rental_monthly_macro_gaps


def build_project_sold_shares(dev):
    """Use only the same project's information through the previous month.

    Return a separate monthly history before the panel join. Do not insert these
    calendar rows into the source used for district-launch aggregation.
    Unknown initial history stays missing, even for eventually sold-out projects.
    """
    import pandas as pd

    months = pd.to_datetime(dev["date"], errors="raise").dt.to_period("M")
    if months.isna().any() or dev["condo_name_key"].isna().any():
        raise ValueError("Project sold-share calculation requires project names and dates")
    keys = pd.MultiIndex.from_arrays(
        [dev["condo_name_key"], months], names=["condo_name_key", "date"],
    )
    if keys.has_duplicates:
        raise ValueError("Duplicate project/month in developer-sales history")
    shares = pd.Series(
        (dev["cumul_sold"] / dev["total_units"]).to_numpy(), index=keys,
    )
    monthly_features = []
    # Include the month after the last report, when that report can first be
    # used as lag1m. Later panel gaps can carry this past-only value forward.
    last_feature_month = months.max() + 1
    for project, values in shares.groupby(level=0, sort=False, observed=True):
        values = values.droplevel(0).sort_index()
        calendar = pd.period_range(values.index.min(), last_feature_month, freq="M")
        current = values.reindex(calendar).ffill()
        monthly = pd.DataFrame({
            "project_sold_share": current,
            "project_sold_share_lag1m": current.shift(1),
        })
        monthly.index = pd.MultiIndex.from_product(
            [[project], calendar], names=["condo_name_key", "date"],
        )
        monthly_features.append(monthly)
    if monthly_features:
        result = pd.concat(monthly_features).reset_index()
        result["date"] = result["date"].dt.to_timestamp()
        return result
    return pd.DataFrame(columns=[
        "condo_name_key", "date", "project_sold_share", "project_sold_share_lag1m",
    ])


def run(settings):
    print("[supply] phase 1")
    import pandas as pd
    import re
    import gc

    SUPPLY_DIR = settings.prep_path("supply")
    settings.folder("02_imputation")
    DEV_SALES_CSV = SUPPLY_DIR / "developer_sales.csv"
    PROJECT_DETAIL_CSV = SUPPLY_DIR / "project_details.csv"
    RENTAL_CSV_PATH = settings.read_imputed("rental")
    SALES_CSV_PATH = settings.read_imputed("sales")
    RENTAL_OUTPUT = settings.folder("03_features") / "rental_supply.csv"
    SALES_OUTPUT = settings.folder("03_features") / "sales_supply.csv"
    EXPECTED_DISTRICTS = set(range(1, 29)) - {24}
    SUPPLY_START = pd.Timestamp("2002-09-01")
    for label, path in {
        "rental imputation input": RENTAL_CSV_PATH,
        "sales imputation input": SALES_CSV_PATH,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    print(f"Rental input : {RENTAL_CSV_PATH}")
    print(f"Sales input  : {SALES_CSV_PATH}")
    print(f"Rental output: {RENTAL_OUTPUT}")
    print(f"Sales output : {SALES_OUTPUT}")
    print("[supply] phase 2")

    print(f"Developer-sales input: {DEV_SALES_CSV}")
    dev = load_supply_table(DEV_SALES_CSV, "developer_sales")
    print(f"Total developer-sales rows: {len(dev):,}")
    print("[supply] phase 3")
    dev.columns = dev.columns.str.strip()
    dev = dev.rename(
        columns={
            "Project Name": "condo_name",
            "Street Name": "street_name",
            "Total Number of Units in Project": "total_units",
            "Units Launched in the Month": "units_launched_month",
            "Units Sold in the Month": "units_sold_month",
            "Cumulative Units Launched to-date": "cumul_launched",
            "Cumulative Units Sold to-date": "cumul_sold",
            "Cumulative Units Launched but Unsold": "cumul_launched_unsold",
            "Total Number of Unsold Units": "total_unsold",
            "Median Price ($ PSF) in the Month": "median_psf",
            "Lowest Price ($ PSF) in the Month": "lowest_psf",
            "Highest Price ($ PSF) in the Month": "highest_psf",
            "Market Segment": "market_segment",
            "Reference Month": "reference_month",
        }
    )
    dev["date"] = pd.to_datetime(dev["reference_month"], format="%b %Y")
    dev["condo_name"] = dev["condo_name"].str.upper().str.strip()
    for col in ["median_psf", "lowest_psf", "highest_psf"]:
        dev[col] = pd.to_numeric(
            dev[col].astype(str).str.replace(",", "").str.strip(), errors="coerce"
        )
    for col in [
        "total_units",
        "units_launched_month",
        "units_sold_month",
        "cumul_launched",
        "cumul_sold",
        "cumul_launched_unsold",
        "total_unsold",
    ]:
        dev[col] = pd.to_numeric(dev[col], errors="coerce")
    dev = dev.drop_duplicates(subset=["condo_name", "date"])
    dev = dev.sort_values(["condo_name", "date"]).reset_index(drop=True)
    print(f"Rows after dedup: {len(dev):,}")
    print(f"Date range: {dev['date'].min()} → {dev['date'].max()}")
    print("[supply] phase 4")

    def normalize_condo_name(name):
        if pd.isna(name):
            return pd.NA
        s = str(name).upper().strip()
        s = s.replace("#", "")
        s = re.sub("\\s*@\\s*", " ", s)
        s = re.sub("\\s+", " ", s).strip()
        return s

    dev["condo_name_key"] = dev["condo_name"].map(normalize_condo_name).astype("string")
    _before = len(dev)
    dev = (
        dev.sort_values(["condo_name_key", "date"])
        .drop_duplicates(["condo_name_key", "date"], keep="last")
        .reset_index(drop=True)
    )
    print(
        f"Developer-sales rows after normalized-key dedup: {len(dev):,} (removed {_before - len(dev):,})"
    )
    print("[supply] phase 5")
    dev = dev.sort_values(["condo_name", "date"]).reset_index(drop=True)
    # F2 fix: no historical backfill from eventual sold-out status.
    project_history = build_project_sold_shares(dev)
    dist_launch = (
        dev.groupby(["district", "date"])["units_launched_month"]
        .sum()
        .reset_index()
        .rename(columns={"units_launched_month": "district_units_launched_month"})
    )
    all_districts = dev["district"].dropna().unique()
    all_months = pd.date_range(dev["date"].min(), dev["date"].max(), freq="MS")
    full_index = pd.MultiIndex.from_product([all_districts, all_months], names=["district", "date"])
    dist_launch = (
        dist_launch.set_index(["district", "date"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["district", "date"])
    )
    dist_launch["district_units_launched_lag1m"] = dist_launch.groupby("district")[
        "district_units_launched_month"
    ].shift(1)
    print("Developer Sales features computed ✓")
    print(
        f"  project_sold_share_lag1m: {project_history['project_sold_share_lag1m'].notna().sum():,} non-null monthly reference rows"
    )
    print(
        f"  district_units_launched_lag1m: {dist_launch['district_units_launched_lag1m'].notna().sum():,} non-null rows"
    )
    print("[supply] phase 6")

    print(f"Project-details input: {PROJECT_DETAIL_CSV}")
    proj = load_supply_table(PROJECT_DETAIL_CSV, "project_details")
    print(f"Total project-detail rows: {len(proj):,}")
    print("[supply] phase 7")
    proj.columns = proj.columns.str.strip()
    proj = proj.rename(
        columns={
            "Project Name": "condo_name",
            "Street Name": "street_name",
            "Developer": "developer",
            "Total Units in Project by Property Type Total Units": "total_units",
            "Total Units in Project by Construction Status Planned": "units_planned",
            "Total Units in Project by Construction Status Under Construction": "units_under_construction",
            "Total Units in Project by Construction Status TOP": "units_top",
            "Total Units in Project by Completion and Sale Status Uncompleted Units Without Pre-Requisites": "units_uncompleted_no_prereq",
            "Total Units in Project by Completion and Sale Status Uncompleted Units With Pre-Requisites Not Launched Yet": "units_not_launched_yet",
            "Total Units in Project by Completion and Sale Status Uncompleted Units With Pre-Requisites Launched": "units_launched",
            "Total Units in Project by Completion and Sale Status Uncompleted Units With Pre-Requisites Sold": "units_sold",
            "Total Units in Project by Completion and Sale Status Uncompleted Units With Pre-Requisites Launched but Unsold": "units_launched_unsold",
            "Total Units in Project by Completion and Sale Status Completed Units With Pre-Requisites Completed and Sold": "units_completed_sold",
            "Total Units in Project by Completion and Sale Status Completed Units With Pre-Requisites Completed but Unsold": "units_completed_unsold",
            "Written Permission Date": "written_permission_date",
        }
    )
    proj["condo_name"] = proj["condo_name"].str.upper().str.strip()
    num_cols = [
        "total_units",
        "units_planned",
        "units_under_construction",
        "units_top",
        "units_uncompleted_no_prereq",
        "units_not_launched_yet",
        "units_launched",
        "units_sold",
        "units_launched_unsold",
        "units_completed_sold",
        "units_completed_unsold",
    ]
    for col in num_cols:
        if col in proj.columns:
            proj[col] = pd.to_numeric(proj[col], errors="coerce")
    quarter_end_month = {1: 3, 2: 6, 3: 9, 4: 12}
    proj["date"] = pd.to_datetime(
        proj["year"].astype(str) + "-" + proj["quarter"].map(quarter_end_month).astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)
    proj = proj.drop_duplicates(subset=["condo_name", "date"])
    print(f"Rows after dedup: {len(proj):,}")
    print(f"Date range: {proj['date'].min()} → {proj['date'].max()}")
    print("[supply] phase 8")
    proj_dedup = proj.drop_duplicates(subset=["condo_name", "district", "date"], keep="last")
    dist_pipeline = (
        proj_dedup.groupby(["district", "date"])
        .agg(
            district_units_under_construction=("units_under_construction", "sum"),
            district_units_not_launched_yet=("units_not_launched_yet", "sum"),
            district_units_launched_unsold=("units_launched_unsold", "sum"),
            district_units_completed_unsold=("units_completed_unsold", "sum"),
        )
        .reset_index()
        .sort_values(["district", "date"])
    )
    pipeline_feat_cols = [
        "district_units_under_construction",
        "district_units_not_launched_yet",
        "district_units_launched_unsold",
        "district_units_completed_unsold",
    ]
    for col in pipeline_feat_cols:
        dist_pipeline[f"{col}_lag1q"] = dist_pipeline.groupby("district")[col].shift(1)
    print("Project Details features computed ✓")
    for col in pipeline_feat_cols:
        n = dist_pipeline[f"{col}_lag1q"].notna().sum()
        print(f"  {col}_lag1q: {n:,} non-null rows")
    print("[supply] phase 9")
    dist_pipeline["date"] = dist_pipeline["date"].dt.to_period("M").dt.to_timestamp()
    min_date = dist_pipeline["date"].min()
    max_date = dist_pipeline["date"].max()
    all_months = pd.date_range(min_date, max_date, freq="MS")
    districts = dist_pipeline["district"].unique()
    monthly_index = pd.MultiIndex.from_product([districts, all_months], names=["district", "date"])
    dist_pipeline_monthly = (
        dist_pipeline.set_index(["district", "date"])
        .reindex(monthly_index)
        .groupby(level="district")
        .ffill()
        .reset_index()
    )
    lag_cols = [f"{c}_lag1q" for c in pipeline_feat_cols]
    print(f"Monthly pipeline rows: {len(dist_pipeline_monthly):,}")
    print(dist_pipeline_monthly[["district", "date"] + lag_cols].head(6))
    print("[supply] phase 10")

    def load_imputed_panel(path, label):
        panel = pd.read_csv(path)
        required = {"project_idx", "date", "condo_name", "district"}
        missing = required - set(panel.columns)
        if missing:
            raise KeyError(f"{label} input missing columns: {sorted(missing)}")
        panel["date"] = pd.to_datetime(panel["date"], errors="raise")
        if label == "Rental":
            panel = fill_rental_monthly_macro_gaps(
                panel, "date", settings.prep_path("macro", "monthly", "sg_macro_monthly.csv"),
                copy=False,
            )
        panel["district"] = pd.to_numeric(panel["district"], errors="raise").astype(int)
        panel["condo_name_key"] = panel["condo_name"].map(normalize_condo_name).astype("string")
        actual_districts = set(panel["district"].unique())
        duplicate_keys = int(panel.duplicated(["project_idx", "date"]).sum())
        print(f"{label} rows       : {len(panel):,}")
        print(f"{label} date range : {panel['date'].min().date()} → {panel['date'].max().date()}")
        print(f"{label} districts  : {sorted(actual_districts)}")
        print(f"{label} duplicate project-date keys: {duplicate_keys:,}")
        assert actual_districts == EXPECTED_DISTRICTS
        assert duplicate_keys == 0
        return panel

    rental = load_imputed_panel(RENTAL_CSV_PATH, "Rental")
    print("[supply] phase 11")
    dev_project = project_history[[
        "condo_name_key", "date", "project_sold_share_lag1m",
    ]].copy()
    dist_launch_merge = dist_launch[["district", "date", "district_units_launched_lag1m"]].copy()
    dist_launch_merge["district"] = pd.to_numeric(dist_launch_merge["district"]).astype(int)
    dist_launch_merge["date"] = (
        pd.to_datetime(dist_launch_merge["date"]).dt.to_period("M").dt.to_timestamp()
    )
    pipeline_merge = dist_pipeline_monthly[["district", "date"] + lag_cols].copy()
    pipeline_merge["district"] = pd.to_numeric(pipeline_merge["district"]).astype(int)
    pipeline_merge["date"] = (
        pd.to_datetime(pipeline_merge["date"]).dt.to_period("M").dt.to_timestamp()
    )
    assert not dev_project.duplicated(["condo_name_key", "date"]).any()
    assert not dist_launch_merge.duplicated(["district", "date"]).any()
    assert not pipeline_merge.duplicated(["district", "date"]).any()
    district_supply_cols = ["district_units_launched_lag1m"] + lag_cols
    all_new_cols = (
        ["project_sold_share_lag1m"] + district_supply_cols + ["is_supply_data_available"]
    )

    def merge_supply_features(panel, label):
        base_rows = len(panel)
        existing = set(all_new_cols) & set(panel.columns)
        if existing:
            raise ValueError(f"{label} already contains supply columns: {sorted(existing)}")
        out = panel.merge(
            dev_project, on=["condo_name_key", "date"], how="left", validate="many_to_one"
        )
        assert len(out) == base_rows, f"{label}: developer merge expanded rows"
        out = out.merge(
            dist_launch_merge, on=["district", "date"], how="left", validate="many_to_one"
        )
        assert len(out) == base_rows, f"{label}: district-launch merge expanded rows"
        out = out.merge(pipeline_merge, on=["district", "date"], how="left", validate="many_to_one")
        assert len(out) == base_rows, f"{label}: pipeline merge expanded rows"
        out["is_supply_data_available"] = (out["date"] >= SUPPLY_START).astype("int8")
        pre_supply = out["is_supply_data_available"].eq(0)
        out.loc[pre_supply, district_supply_cols] = out.loc[
            pre_supply, district_supply_cols
        ].fillna(0)
        print(f"\n{label} merge rows preserved: {len(out):,}")
        for col in all_new_cols:
            print(f"  {col}: {100 * out[col].notna().mean():.1f}% non-null")
        matched_projects = out.loc[
            out["project_sold_share_lag1m"].notna(), "condo_name_key"
        ].nunique()
        total_projects = out["condo_name_key"].nunique()
        print(f"  project supply match: {matched_projects:,}/{total_projects:,} condos")
        return out

    result = merge_supply_features(rental, "Rental")
    print("[supply] phase 12")
    print("=== Rental supply features summary ===")
    display(result[all_new_cols].describe(include="all").T)
    print("[supply] phase 13")
    rental_out = result.drop(columns=["condo_name_key"])
    assert len(rental_out) == len(rental)
    rental_out.to_csv(RENTAL_OUTPUT, index=False)
    print(f"Saved rental → {RENTAL_OUTPUT}")
    print(f"Shape: {rental_out.shape}")
    del rental, result, rental_out
    gc.collect()
    print("Released rental panel from memory.")
    print("[supply] phase 14")
    sales = load_imputed_panel(SALES_CSV_PATH, "Sales")
    sales_result = merge_supply_features(sales, "Sales")
    sales_out = sales_result.drop(columns=["condo_name_key"])
    assert len(sales_out) == len(sales)
    sales_out.to_csv(SALES_OUTPUT, index=False)
    print(f"\nSaved sales → {SALES_OUTPUT}")
    print(f"Shape: {sales_out.shape}")
    print("=== Sales supply features summary ===")
    display(sales_out[all_new_cols].describe(include="all").T)
    print("[supply] phase 15")
    print("Supply merging completed for both full-district panels.")
    print(f"Rental output exists: {RENTAL_OUTPUT.exists()}")
    print(f"Sales output exists : {SALES_OUTPUT.exists()}")
    return None
