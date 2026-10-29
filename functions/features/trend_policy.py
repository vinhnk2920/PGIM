"""trend_policy. Explicit callable; no work is performed on import."""

from functions.common import display


def run(settings):
    print("[trend_policy] phase 1")
    import gc
    import numpy as np
    import pandas as pd

    settings.folder("03_features")
    TREND_POLICY_DIR = settings.prep_path("trend_policy")
    RENT_INPUT = settings.folder("03_features") / "rental_consulting.csv"
    SALES_INPUT = settings.folder("03_features") / "sales_consulting.csv"
    RENT_EXT_PATH = TREND_POLICY_DIR / "condo_rent_external_features_consolidated.csv"
    SALES_EXT_PATH = TREND_POLICY_DIR / "condo_sales_external_features_consolidated.csv"
    RENT_OUTPUT = settings.folder("03_features") / "rental_trend_policy.csv"
    SALES_OUTPUT = settings.folder("03_features") / "sales_trend_policy.csv"
    RENT_FEATURES = [
        "gt_condo_rental_singapore_ma3",
        "gt_hdb_rent_singapore_ma3",
        "gt_singapore_international_schools_ma3",
        "gt_cost_of_living_singapore_ma3",
        "absd_investor_avg_level",
        "absd_event_3m",
        "resi_composite_index",
        "resi_future_index",
        "gls_resi_num_sites_awarded_36m",
        "gls_resi_total_gfa_awarded_36m",
    ]
    SALES_FEATURES = [
        "gt_condo_sale_ma3",
        "gt_new_condo_launch_singapore_ma3",
        "gt_hdb_resale_singapore_ma3",
        "gt_en_bloc_singapore_ma3",
        "gt_cost_of_living_singapore_ma3",
        "absd_investor_avg_level",
        "absd_event_3m",
        "resi_composite_index",
        "resi_future_index",
        "gls_resi_total_gfa_awarded_36m",
        "gls_resi_median_land_price_psm_gfa_12m",
        "gls_resi_avg_num_bids_12m",
        "gls_resi_num_sites_awarded_36m",
    ]
    CHUNK_SIZE = 100000
    EXPECTED_DISTRICTS = set(range(1, 29)) - {24}
    required_files = [RENT_INPUT, SALES_INPUT, RENT_EXT_PATH, SALES_EXT_PATH]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing_files))
    print("Rental input:", RENT_INPUT)
    print("Sales input:", SALES_INPUT)
    print("Rental output:", RENT_OUTPUT)
    print("Sales output:", SALES_OUTPUT)
    print("[trend_policy] phase 2")

    def load_external_features(path, expected_features, label):
        external = pd.read_csv(path)
        expected_columns = {"month", *expected_features}
        missing = expected_columns - set(external.columns)
        unexpected = set(external.columns) - expected_columns
        if missing:
            raise KeyError(f"{label}: missing external columns {sorted(missing)}")
        if unexpected:
            raise ValueError(f"{label}: unexpected external columns {sorted(unexpected)}")
        external["date"] = (
            pd.to_datetime(external["month"], format="%m/%d/%Y", errors="coerce")
            .dt.to_period("M")
            .dt.start_time
        )
        if external["date"].isna().any():
            raise ValueError(f"{label}: {int(external['date'].isna().sum()):,} invalid months")
        if external["date"].duplicated().any():
            duplicates = external.loc[external["date"].duplicated(False), "date"].tolist()
            raise AssertionError(f"{label}: duplicate months {duplicates[:10]}")
        numeric = external[expected_features].apply(pd.to_numeric, errors="coerce")
        newly_invalid = numeric.isna() & external[expected_features].notna()
        if newly_invalid.any().any():
            bad_columns = newly_invalid.columns[newly_invalid.any()].tolist()
            raise ValueError(f"{label}: non-numeric values in {bad_columns}")
        external[expected_features] = numeric
        if np.isinf(external[expected_features].to_numpy(dtype=float)).any():
            raise ValueError(f"{label}: infinite feature values found")
        external = external[["date"] + expected_features].sort_values("date").reset_index(drop=True)
        feature_windows = []
        for col in expected_features:
            valid = external.loc[external[col].notna(), "date"]
            feature_windows.append(
                {
                    "feature": col,
                    "first_valid": valid.min() if len(valid) else pd.NaT,
                    "last_valid": valid.max() if len(valid) else pd.NaT,
                    "missing_months": int(external[col].isna().sum()),
                }
            )
        print(
            f"{label}: {len(external)} unique months, {external['date'].min().date()} to {external['date'].max().date()}"
        )
        display(pd.DataFrame(feature_windows))
        return external

    rent_external = load_external_features(RENT_EXT_PATH, RENT_FEATURES, "Rental external")
    sales_external = load_external_features(SALES_EXT_PATH, SALES_FEATURES, "Sales external")
    print("\nPoint-in-time prerequisite:")
    print("- Google Trends *_ma3 columns must be trailing MA(3), not centered MA(3).")
    print("- RESI values must use the documented publication-availability lag.")
    print("- GLS rolling windows must be based on award dates no later than the feature month.")
    print("[trend_policy] phase 3")

    def merge_external_features(
        input_path, output_path, external, feature_cols, label, chunk_size=CHUNK_SIZE
    ):
        print(f"\n--- {label} ---")
        partial_path = output_path.parent / f"{output_path.name}.partial"
        total_input_rows = 0
        total_output_rows = 0
        input_column_count = None
        actual_districts = set()
        duplicate_keys = 0
        previous_last_key = None
        non_null_counts = {col: 0 for col in feature_cols}
        first_valid = {col: pd.NaT for col in feature_cols}
        last_valid = {col: pd.NaT for col in feature_cols}
        lookup = external[["date"] + feature_cols]
        if lookup["date"].duplicated().any():
            raise AssertionError(f"{label}: external lookup has duplicate months")
        for chunk_number, frame in enumerate(
            pd.read_csv(input_path, chunksize=chunk_size, low_memory=False), start=1
        ):
            if chunk_number == 1:
                input_column_count = len(frame.columns)
                required = {"project_idx", "date", "district"}
                missing = required - set(frame.columns)
                if missing:
                    raise KeyError(f"{label}: missing input columns {sorted(missing)}")
                collisions = sorted(set(frame.columns) & set(feature_cols))
                if collisions:
                    raise ValueError(f"{label}: feature columns already exist: {collisions}")
            rows_this_chunk = len(frame)
            total_input_rows += rows_this_chunk
            frame["date"] = (
                pd.to_datetime(frame["date"], errors="coerce").dt.to_period("M").dt.start_time
            )
            if frame["date"].isna().any():
                raise ValueError(f"{label}, chunk {chunk_number}: invalid dates found")
            district_numeric = pd.to_numeric(frame["district"], errors="coerce")
            actual_districts.update(district_numeric.dropna().astype(int).unique())
            keys = pd.MultiIndex.from_frame(frame[["project_idx", "date"]])
            if not keys.is_monotonic_increasing:
                raise AssertionError(
                    f"{label}: project_idx/date is not sorted in chunk {chunk_number}"
                )
            duplicate_keys += int(keys.duplicated().sum())
            if len(keys) and previous_last_key == keys[0]:
                duplicate_keys += 1
            if len(keys):
                previous_last_key = keys[-1]
            merged = frame.merge(lookup, on="date", how="left", validate="many_to_one")
            if len(merged) != rows_this_chunk:
                raise AssertionError(f"{label}, chunk {chunk_number}: row count changed")
            if merged.shape[1] != input_column_count + len(feature_cols):
                raise AssertionError(f"{label}: expected exactly {len(feature_cols)} new columns")
            if np.isinf(merged[feature_cols].to_numpy(dtype=float)).any():
                raise ValueError(f"{label}, chunk {chunk_number}: infinite merged values")
            for col in feature_cols:
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
                partial_path,
                mode="w" if chunk_number == 1 else "a",
                header=chunk_number == 1,
                index=False,
            )
            total_output_rows += len(merged)
            if chunk_number == 1 or chunk_number % 5 == 0:
                print(f"  processed {total_output_rows:,} rows")
            del frame, merged, keys
            gc.collect()
        if total_input_rows != total_output_rows:
            raise AssertionError(
                f"{label}: total row count changed from {total_input_rows:,} to {total_output_rows:,}"
            )
        if duplicate_keys:
            raise AssertionError(f"{label}: {duplicate_keys:,} duplicate project-month keys")
        missing_districts = EXPECTED_DISTRICTS - actual_districts
        unexpected_districts = actual_districts - EXPECTED_DISTRICTS
        partial_path.replace(output_path)
        coverage = pd.DataFrame(
            {
                "feature": feature_cols,
                "non_null_rows": [non_null_counts[col] for col in feature_cols],
                "coverage_pct": [
                    100 * non_null_counts[col] / total_output_rows for col in feature_cols
                ],
                "first_valid_date": [first_valid[col] for col in feature_cols],
                "last_valid_date": [last_valid[col] for col in feature_cols],
            }
        )
        summary = {
            "dataset": label,
            "input_rows": total_input_rows,
            "output_rows": total_output_rows,
            "input_columns": input_column_count,
            "features_added": len(feature_cols),
            "output_columns": input_column_count + len(feature_cols),
            "district_count": len(actual_districts),
            "missing_expected_districts": sorted(missing_districts),
            "unexpected_districts": sorted(unexpected_districts),
            "duplicate_project_months": duplicate_keys,
            "output_path": str(output_path),
        }
        print(f"Rows: {total_input_rows:,} -> {total_output_rows:,} (unchanged)")
        print(
            f"Columns: {input_column_count} + {len(feature_cols)} -> {input_column_count + len(feature_cols)}"
        )
        print(
            f"Districts: {len(actual_districts)}; missing: {sorted(missing_districts)}; unexpected: {sorted(unexpected_districts)}"
        )
        print("Saved:", output_path)
        display(coverage)
        gc.collect()
        return (summary, coverage)

    print("[trend_policy] phase 4")
    rental_summary, rental_coverage = merge_external_features(
        RENT_INPUT, RENT_OUTPUT, rent_external, RENT_FEATURES, "Rental"
    )
    gc.collect()
    sales_summary, sales_coverage = merge_external_features(
        SALES_INPUT, SALES_OUTPUT, sales_external, SALES_FEATURES, "Sales"
    )
    print("[trend_policy] phase 5")
    summary = pd.DataFrame([rental_summary, sales_summary])
    display(summary)
    for path in [RENT_OUTPUT, SALES_OUTPUT]:
        print(
            path.name,
            "exists:",
            path.exists(),
            "size (MB):",
            round(path.stat().st_size / 1024**2, 1) if path.exists() else None,
        )
    print("\nCoverage:")
    display(
        pd.concat(
            [rental_coverage.assign(dataset="Rental"), sales_coverage.assign(dataset="Sales")],
            ignore_index=True,
        )
    )
    return None
