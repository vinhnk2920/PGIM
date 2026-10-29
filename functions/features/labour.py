"""labour. Explicit callable; no work is performed on import."""

from functions.common import display
from functions.features.panel_io import read_panel_chunks


def run(settings):
    print("[labour] phase 1")
    import gc
    import numpy as np
    import pandas as pd

    MOM_DIR = settings.prep_path("labour")
    RET_PATH = MOM_DIR / "ret_01_total_qtr.csv"
    RCR_PATH = MOM_DIR / "rcr_01_total_qtr.csv"
    RGR_PATH = MOM_DIR / "rgr_01_total_qtr.xlsx"
    INCOME_PATH = MOM_DIR / "median_gross_monthly_income_full_time_employed_residents_total.csv"
    EP_PATH = MOM_DIR / "share_nonresident_population_by_pass_types_annual.csv"
    RENTAL_INPUT = settings.folder("03_features") / "rental_supply.csv"
    SALES_INPUT = settings.folder("03_features") / "sales_supply.csv"
    RENTAL_OUTPUT = settings.folder("03_features") / "rental_labour.csv"
    SALES_OUTPUT = settings.folder("03_features") / "sales_labour.csv"
    required_files = [RET_PATH, RCR_PATH, RGR_PATH, INCOME_PATH, EP_PATH, RENTAL_INPUT, SALES_INPUT]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing_files))
    EXPECTED_DISTRICTS = set(range(1, 29)) - {24}
    print("MOM directory:", MOM_DIR)
    print("Rental input:", RENTAL_INPUT)
    print("Sales input:", SALES_INPUT)
    print("[labour] phase 2")
    ret = pd.read_csv(RET_PATH)
    ret = ret.rename(columns={ret.columns[0]: "quarter", ret.columns[1]: "retrenchment_total"})
    ret = ret[["quarter", "retrenchment_total"]].copy()
    ret["retrenchment_total"] = pd.to_numeric(ret["retrenchment_total"], errors="coerce")
    ret["date"] = pd.PeriodIndex(ret["quarter"].astype(str).str.strip(), freq="Q").to_timestamp()
    ret = ret.sort_values("date").drop_duplicates("date", keep="last")
    ret["retrenchment_total_lag1q"] = ret["retrenchment_total"].shift(1)
    ret["retrenchment_yoy_pct"] = ret["retrenchment_total"].pct_change(4, fill_method=None) * 100
    ret["retrenchment_yoy_pct_lag1q"] = ret["retrenchment_yoy_pct"].shift(1)
    ret = ret.replace([np.inf, -np.inf], np.nan)
    print("[labour] phase 3")
    rcr = pd.read_csv(RCR_PATH)
    rcr = rcr.rename(columns={rcr.columns[0]: "quarter", rcr.columns[1]: "recruitment_rate"})
    rcr = rcr[["quarter", "recruitment_rate"]].copy()
    rcr["recruitment_rate"] = pd.to_numeric(rcr["recruitment_rate"], errors="coerce")
    rcr["date"] = pd.PeriodIndex(rcr["quarter"].astype(str).str.strip(), freq="Q").to_timestamp()
    rcr = rcr.sort_values("date").drop_duplicates("date", keep="last")
    rcr["recruitment_rate_lag1q"] = rcr["recruitment_rate"].shift(1)
    print("[labour] phase 4")
    rgr = pd.read_excel(RGR_PATH, engine="openpyxl")
    rgr = rgr.rename(columns={rgr.columns[0]: "quarter", rgr.columns[1]: "resignation_rate"})
    rgr = rgr[["quarter", "resignation_rate"]].copy()
    rgr["resignation_rate"] = pd.to_numeric(rgr["resignation_rate"], errors="coerce")
    rgr["date"] = pd.PeriodIndex(rgr["quarter"].astype(str).str.strip(), freq="Q").to_timestamp()
    rgr = rgr.sort_values("date").drop_duplicates("date", keep="last")
    rgr["resignation_rate_lag1q"] = rgr["resignation_rate"].shift(1)
    print("[labour] phase 5")

    def quarterly_to_monthly(df, date_col, value_cols):
        q = df[[date_col] + value_cols].copy().sort_values(date_col)
        q[date_col] = pd.to_datetime(q[date_col]).dt.to_period("Q").dt.start_time
        q = q.drop_duplicates(date_col, keep="last")
        monthly_index = pd.date_range(
            q[date_col].min(), q[date_col].max() + pd.offsets.MonthBegin(2), freq="MS"
        )
        monthly = q.set_index(date_col)[value_cols].reindex(monthly_index).ffill()
        monthly.index.name = "date"
        return monthly.reset_index()

    ret_m = quarterly_to_monthly(
        ret, "date", ["retrenchment_total_lag1q", "retrenchment_yoy_pct_lag1q"]
    )
    rcr_m = quarterly_to_monthly(rcr, "date", ["recruitment_rate_lag1q"])
    rgr_m = quarterly_to_monthly(rgr, "date", ["resignation_rate_lag1q"])
    print("Quarterly monthly endpoints:")
    print("  retrenchment:", ret_m["date"].min().date(), "to", ret_m["date"].max().date())
    print("  recruitment:", rcr_m["date"].min().date(), "to", rcr_m["date"].max().date())
    print("  resignation:", rgr_m["date"].min().date(), "to", rgr_m["date"].max().date())
    print("[labour] phase 6")
    income = pd.read_csv(INCOME_PATH)
    income = income[["year", "median_income_incl_emp_cpf"]].rename(
        columns={"median_income_incl_emp_cpf": "median_income"}
    )
    income["year"] = pd.to_numeric(income["year"], errors="coerce")
    income["median_income"] = pd.to_numeric(income["median_income"], errors="coerce")
    income = income.dropna(subset=["year"]).sort_values("year").drop_duplicates("year", keep="last")
    annual_frame = pd.DataFrame({"year": range(2000, 2027)})
    income = annual_frame.merge(income, on="year", how="left", validate="one_to_one").sort_values(
        "year"
    )
    income["median_income"] = income["median_income"].ffill()
    income["median_income_yoy_pct"] = income["median_income"].pct_change(fill_method=None) * 100
    income["median_income_lag1y"] = income["median_income"].shift(1)
    income["median_income_yoy_pct_lag"] = income["median_income_yoy_pct"].shift(1)
    income["date"] = pd.to_datetime(income["year"].astype(str) + "-01-01")
    all_months = pd.date_range("2000-01-01", "2026-12-01", freq="MS")
    income_m = (
        income.set_index("date")[["median_income_lag1y", "median_income_yoy_pct_lag"]]
        .reindex(all_months)
        .ffill()
        .rename_axis("date")
        .reset_index()
    )
    print("[labour] phase 7")
    print(
        "Retrenchment quarterly rows:",
        len(ret),
        "range:",
        ret["date"].min().date(),
        "to",
        ret["date"].max().date(),
    )
    print(
        "Recruitment quarterly rows:",
        len(rcr),
        "range:",
        rcr["date"].min().date(),
        "to",
        rcr["date"].max().date(),
    )
    print(
        "Resignation quarterly rows:",
        len(rgr),
        "range:",
        rgr["date"].min().date(),
        "to",
        rgr["date"].max().date(),
    )
    print(
        "Income annual frame rows:",
        len(income),
        "range:",
        int(income["year"].min()),
        "to",
        int(income["year"].max()),
    )
    print("[labour] phase 8")
    ep_raw = pd.read_csv(EP_PATH)
    if "DataSeries" not in ep_raw.columns or "_id" not in ep_raw.columns:
        raise ValueError(f"Unexpected pass-share schema: {list(ep_raw.columns)}")
    ep_wide = ep_raw.set_index("DataSeries").drop(columns=["_id"])
    ep_wide.columns = ep_wide.columns.astype(str)
    ep_series = pd.to_numeric(
        ep_wide.loc["Employment Pass Holders"].replace("na", np.nan), errors="coerce"
    )
    sp_series = pd.to_numeric(ep_wide.loc["S Pass Holders"].replace("na", np.nan), errors="coerce")
    ep = pd.DataFrame(
        {
            "year": ep_series.index.astype(int),
            "ep_share_pct": ep_series.to_numpy(),
            "sp_share_pct": sp_series.to_numpy(),
        }
    ).sort_values("year")
    annual_frame_ep = pd.DataFrame({"year": range(2000, 2027)})
    ep = annual_frame_ep.merge(ep, on="year", how="left", validate="one_to_one").sort_values("year")
    ep["ep_share_pct"] = ep["ep_share_pct"].ffill()
    ep["sp_share_pct"] = ep["sp_share_pct"].ffill()
    ep["ep_share_pct_lag1y"] = ep["ep_share_pct"].shift(1)
    ep["sp_share_pct_lag1y"] = ep["sp_share_pct"].shift(1)
    ep["date"] = pd.to_datetime(ep["year"].astype(str) + "-01-01")
    ep_m = (
        ep.set_index("date")[["ep_share_pct_lag1y", "sp_share_pct_lag1y"]]
        .reindex(all_months)
        .ffill()
        .rename_axis("date")
        .reset_index()
    )
    print("[labour] phase 9")
    print(
        "Foreign workforce annual rows:",
        len(ep),
        "range:",
        int(ep["year"].min()),
        "to",
        int(ep["year"].max()),
    )
    print("First non-null EP-share year:", ep.loc[ep["ep_share_pct"].notna(), "year"].min())
    print("First non-null S-Pass-share year:", ep.loc[ep["sp_share_pct"].notna(), "year"].min())
    print("[labour] phase 10")
    MOM_FEATURES = [
        "retrenchment_total_lag1q",
        "retrenchment_yoy_pct_lag1q",
        "recruitment_rate_lag1q",
        "resignation_rate_lag1q",
        "median_income_lag1y",
        "median_income_yoy_pct_lag",
        "ep_share_pct_lag1y",
        "sp_share_pct_lag1y",
    ]
    mom = ret_m.copy()
    for frame in [rcr_m, rgr_m, income_m, ep_m]:
        mom = mom.merge(frame, on="date", how="outer", validate="one_to_one")
    mom = mom.sort_values("date").replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    if mom["date"].duplicated().any():
        raise AssertionError("MOM table contains duplicate months")
    feature_windows = []
    for col in MOM_FEATURES:
        valid = mom.loc[mom[col].notna(), "date"]
        feature_windows.append(
            {
                "feature": col,
                "first_valid": valid.min() if len(valid) else pd.NaT,
                "last_valid": valid.max() if len(valid) else pd.NaT,
                "non_null_months": int(mom[col].notna().sum()),
            }
        )
    feature_windows = pd.DataFrame(feature_windows)
    print("MOM monthly range:", mom["date"].min().date(), "to", mom["date"].max().date())
    display(feature_windows)
    print("[labour] phase 11")

    def merge_mom_file(input_path, output_path, label, chunk_size=100000):
        print(f"\n--- {label} ---")
        partial_path = output_path.parent / f"{output_path.name}.partial"
        total_input_rows = 0
        total_output_rows = 0
        base_cols = None
        actual_districts = set()
        duplicated_keys = 0
        previous_last_key = None
        non_null_counts = {col: 0 for col in MOM_FEATURES}
        first_valid = {col: pd.NaT for col in MOM_FEATURES}
        last_valid = {col: pd.NaT for col in MOM_FEATURES}
        for chunk_no, df in enumerate(read_panel_chunks(input_path, chunk_size), start=1):
            if chunk_no == 1:
                base_cols = len(df.columns)
                required = {"project_idx", "date", "district"}
                missing = required - set(df.columns)
                if missing:
                    raise KeyError(f"{label}: missing required columns {sorted(missing)}")
                overlapping = sorted(set(MOM_FEATURES) & set(df.columns))
                if overlapping:
                    raise ValueError(f"{label}: MOM columns already exist: {overlapping}")
            rows_this_chunk = len(df)
            total_input_rows += rows_this_chunk
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.start_time
            if df["date"].isna().any():
                raise ValueError(
                    f"{label}, chunk {chunk_no}: {int(df['date'].isna().sum()):,} invalid dates"
                )
            district_num = pd.to_numeric(df["district"], errors="coerce")
            actual_districts.update(district_num.dropna().astype(int).unique())
            keys = pd.MultiIndex.from_frame(df[["project_idx", "date"]])
            if not keys.is_monotonic_increasing:
                raise AssertionError(f"{label}: project_idx/date is not sorted in chunk {chunk_no}")
            duplicated_keys += int(keys.duplicated().sum())
            if len(keys) and previous_last_key == keys[0]:
                duplicated_keys += 1
            if len(keys):
                previous_last_key = keys[-1]
            merged = df.merge(
                mom[["date"] + MOM_FEATURES], on="date", how="left", validate="many_to_one"
            )
            if len(merged) != rows_this_chunk:
                raise AssertionError(f"{label}, chunk {chunk_no}: row count changed during merge")
            if merged.shape[1] != base_cols + len(MOM_FEATURES):
                raise AssertionError(f"{label}: expected exactly {len(MOM_FEATURES)} new columns")
            for col in MOM_FEATURES:
                valid_dates = merged.loc[merged[col].notna(), "date"]
                non_null_counts[col] += int(len(valid_dates))
                if len(valid_dates):
                    chunk_first, chunk_last = (valid_dates.min(), valid_dates.max())
                    first_valid[col] = (
                        chunk_first
                        if pd.isna(first_valid[col])
                        else min(first_valid[col], chunk_first)
                    )
                    last_valid[col] = (
                        chunk_last if pd.isna(last_valid[col]) else max(last_valid[col], chunk_last)
                    )
            merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
            merged.to_csv(
                partial_path, mode="w" if chunk_no == 1 else "a", header=chunk_no == 1, index=False
            )
            total_output_rows += len(merged)
            if chunk_no == 1 or chunk_no % 5 == 0:
                print(f"  processed {total_output_rows:,} rows")
            del df, merged, keys
            gc.collect()
        if total_input_rows != total_output_rows:
            raise AssertionError(
                f"{label}: total row count changed from {total_input_rows:,} to {total_output_rows:,}"
            )
        if duplicated_keys:
            raise AssertionError(
                f"{label}: found {duplicated_keys:,} duplicate project_idx-date keys"
            )
        missing_districts = EXPECTED_DISTRICTS - actual_districts
        unexpected_districts = actual_districts - EXPECTED_DISTRICTS
        partial_path.replace(output_path)
        coverage = pd.DataFrame(
            {
                "feature": MOM_FEATURES,
                "non_null_rows": [non_null_counts[col] for col in MOM_FEATURES],
                "coverage_pct": [
                    100 * non_null_counts[col] / total_output_rows for col in MOM_FEATURES
                ],
                "first_valid_input_date": [first_valid[col] for col in MOM_FEATURES],
                "last_valid_input_date": [last_valid[col] for col in MOM_FEATURES],
            }
        )
        print(f"Input/output rows: {total_input_rows:,} (unchanged)")
        print(
            f"Districts: {len(actual_districts)}; missing expected: {sorted(missing_districts)}; unexpected: {sorted(unexpected_districts)}"
        )
        print("Saved:", output_path)
        display(coverage)
        summary = {
            "dataset": label,
            "input_rows": total_input_rows,
            "output_rows": total_output_rows,
            "input_columns": base_cols,
            "output_columns": base_cols + len(MOM_FEATURES),
            "district_count": len(actual_districts),
            "missing_expected_districts": sorted(missing_districts),
            "duplicate_project_months": duplicated_keys,
            "output_path": str(output_path),
        }
        gc.collect()
        return (summary, coverage)

    rental_summary, rental_coverage = merge_mom_file(RENTAL_INPUT, RENTAL_OUTPUT, "Rental")
    sales_summary, sales_coverage = merge_mom_file(SALES_INPUT, SALES_OUTPUT, "Sales")
    print("[labour] phase 12")
    pd.DataFrame([rental_summary, sales_summary])
    print("[labour] phase 13")
    (
        rental_coverage[["feature", "coverage_pct"]]
        .rename(columns={"coverage_pct": "rental_coverage_pct"})
        .merge(
            sales_coverage[["feature", "coverage_pct"]].rename(
                columns={"coverage_pct": "sales_coverage_pct"}
            ),
            on="feature",
            validate="one_to_one",
        )
    )
    print("[labour] phase 14")
    for path in [RENTAL_OUTPUT, SALES_OUTPUT]:
        print(
            path.name,
            "exists:",
            path.exists(),
            "size (MB):",
            round(path.stat().st_size / 1024**2, 1) if path.exists() else None,
        )
    return None
