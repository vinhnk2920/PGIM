"""nearby_prices. Explicit callable; no work is performed on import."""

from functions.common import display


def run(settings):
    print("[nearby_prices] phase 1")
    import gc
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import BallTree

    DATA_DIR = settings.folder("03_features")
    INPUT_DIR = settings.folder("03_features")
    RENT_INPUT = settings.folder("03_features") / "rental_trend_policy.csv"
    SALES_INPUT = settings.folder("03_features") / "sales_trend_policy.csv"
    RENT_FEATURE_OUTPUT = (
        DATA_DIR / "rental_nearby_price_features_by_condo_month_full_districts.csv"
    )
    SALES_FEATURE_OUTPUT = (
        DATA_DIR / "sales_nearby_price_features_by_condo_month_full_districts.csv"
    )
    RENT_OUTPUT = settings.model_input("rental")
    SALES_OUTPUT = settings.model_input("sales")
    RADII_KM = (0.5, 1.0)
    WINDOWS_MONTHS = (6, 12)
    MOMENTUM_MONTHS = 12
    EARTH_RADIUS_KM = 6371.0088
    MIN_DISTANCE_KM = 0.05
    READ_CHUNK_SIZE = 200000
    WRITE_CHUNK_SIZE = 100000
    EXPECTED_DISTRICTS = set(range(1, 29)) - {24}
    for path in [RENT_INPUT, SALES_INPUT]:
        if not path.exists():
            raise FileNotFoundError(path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Rental input:", RENT_INPUT)
    print("Sales input:", SALES_INPUT)
    print("Compact feature directory:", DATA_DIR)
    print("[nearby_prices] phase 2")

    def normalize_condo(series):
        return series.astype("string").str.strip().str.replace("\\s+", " ", regex=True).str.upper()

    def radius_tag(radius_km):
        return f"{int(round(radius_km * 1000))}m"

    def load_panel_core_chunked(input_path, observed_price_col, label):
        required = ["condo_name", "date", "latitude", "longitude", observed_price_col]
        available = pd.read_csv(input_path, nrows=0).columns.tolist()
        missing = sorted(set(required) - set(available))
        if missing:
            raise KeyError(f"{label}: missing required columns {missing}")
        panel_rows = 0
        observed_rows = 0
        min_month = None
        max_month = None
        observed_parts = []
        coordinate_parts = []
        for chunk_number, frame in enumerate(
            pd.read_csv(input_path, usecols=required, chunksize=READ_CHUNK_SIZE, low_memory=False),
            start=1,
        ):
            condo_key = normalize_condo(frame["condo_name"])
            month_key = (
                pd.to_datetime(frame["date"], errors="coerce").dt.to_period("M").dt.start_time
            )
            price = pd.to_numeric(frame[observed_price_col], errors="coerce")
            if condo_key.isna().any():
                raise ValueError(f"{label}, chunk {chunk_number}: missing condo names")
            if month_key.isna().any():
                raise ValueError(f"{label}, chunk {chunk_number}: invalid dates")
            if frame[["latitude", "longitude"]].isna().any().any():
                raise ValueError(f"{label}, chunk {chunk_number}: missing coordinates")
            if (
                not frame["latitude"].between(-90, 90).all()
                or not frame["longitude"].between(-180, 180).all()
            ):
                raise ValueError(f"{label}, chunk {chunk_number}: invalid coordinate ranges")
            helper = pd.DataFrame(
                {
                    "_condo_key": condo_key,
                    "_month_key": month_key,
                    "latitude": pd.to_numeric(frame["latitude"], errors="coerce"),
                    "longitude": pd.to_numeric(frame["longitude"], errors="coerce"),
                    "_observed_price": price,
                }
            )
            coordinate_parts.append(
                helper[["_condo_key", "latitude", "longitude"]].drop_duplicates()
            )
            observed = helper.loc[
                helper["_observed_price"].notna(), ["_month_key", "_condo_key", "_observed_price"]
            ]
            if len(observed):
                observed_parts.append(observed)
            panel_rows += len(frame)
            observed_rows += len(observed)
            chunk_min, chunk_max = (month_key.min(), month_key.max())
            min_month = chunk_min if min_month is None else min(min_month, chunk_min)
            max_month = chunk_max if max_month is None else max(max_month, chunk_max)
            if chunk_number == 1 or chunk_number % 5 == 0:
                print(f"  {label}: scanned {panel_rows:,} panel rows")
            del frame, helper, observed
            gc.collect()
        coordinates_raw = pd.concat(coordinate_parts, ignore_index=True).drop_duplicates()
        coordinate_check = coordinates_raw.groupby("_condo_key").agg(
            latitude_min=("latitude", "min"),
            latitude_max=("latitude", "max"),
            longitude_min=("longitude", "min"),
            longitude_max=("longitude", "max"),
        )
        conflicts = coordinate_check[
            ~np.isclose(
                coordinate_check["latitude_min"], coordinate_check["latitude_max"], atol=1e-08
            )
            | ~np.isclose(
                coordinate_check["longitude_min"], coordinate_check["longitude_max"], atol=1e-08
            )
        ]
        if len(conflicts):
            raise ValueError(
                f"{label}: {len(conflicts)} normalized condo names have conflicting coordinates"
            )
        coordinates = (
            coordinates_raw.groupby("_condo_key", as_index=False)[["latitude", "longitude"]]
            .median()
            .sort_values("_condo_key")
            .reset_index(drop=True)
        )
        observations = pd.concat(observed_parts, ignore_index=True)
        if np.isinf(observations["_observed_price"].to_numpy(dtype=float)).any():
            raise ValueError(f"{label}: infinite observed prices")
        print(
            f"{label}: {panel_rows:,} panel rows, {len(coordinates):,} condos, {observed_rows:,} observed-price rows, {min_month.date()} to {max_month.date()}"
        )
        return (coordinates, observations, min_month, max_month, panel_rows, observed_rows)

    def trailing_stats_for_focal(price_matrix, neighbor_idx, neighbor_distance_km, window):
        n_months = price_matrix.shape[0]
        median = np.full(n_months, np.nan, dtype="float64")
        weighted_mean = np.full(n_months, np.nan, dtype="float64")
        observation_count = np.zeros(n_months, dtype="int32")
        condo_count = np.zeros(n_months, dtype="int16")
        if len(neighbor_idx) == 0:
            return (median, weighted_mean, observation_count, condo_count)
        neighbor_prices = price_matrix[:, neighbor_idx]
        weights = 1.0 / np.maximum(neighbor_distance_km, MIN_DISTANCE_KM)
        for t in range(1, n_months):
            start = max(0, t - window)
            block = neighbor_prices[start:t, :]
            valid = np.isfinite(block)
            n_valid = int(valid.sum())
            observation_count[t] = n_valid
            if n_valid == 0:
                continue
            median[t] = np.nanmedian(block)
            denominator = np.sum(valid * weights[None, :])
            weighted_mean[t] = np.nansum(block * weights[None, :]) / denominator
            condo_count[t] = int(valid.any(axis=0).sum())
        return (median, weighted_mean, observation_count, condo_count)

    def build_nearby_feature_table(input_path, observed_price_col, label):
        coordinates, observations, min_month, max_month, panel_rows, observed_rows = (
            load_panel_core_chunked(input_path, observed_price_col, label)
        )
        condo_keys = coordinates["_condo_key"].tolist()
        months = pd.date_range(min_month, max_month, freq="MS")
        condo_month = (
            observations.groupby(["_month_key", "_condo_key"], observed=True)["_observed_price"]
            .median()
            .unstack("_condo_key")
            .reindex(index=months, columns=condo_keys)
        )
        price_matrix = condo_month.to_numpy(dtype="float64")
        observed_matrix = np.isfinite(price_matrix)
        known_prior = np.zeros_like(observed_matrix, dtype=bool)
        if len(months) > 1:
            known_prior[1:, :] = np.maximum.accumulate(observed_matrix[:-1, :], axis=0)
        coordinates_rad = np.deg2rad(
            coordinates[["latitude", "longitude"]].to_numpy(dtype="float64")
        )
        tree = BallTree(coordinates_rad, metric="haversine")
        n_months, n_condos = price_matrix.shape
        feature_arrays = {}
        for radius_km in RADII_KM:
            tag = radius_tag(radius_km)
            indices, distances_rad = tree.query_radius(
                coordinates_rad,
                r=radius_km / EARTH_RADIUS_KM,
                return_distance=True,
                sort_results=True,
            )
            neighbor_indices = []
            neighbor_distances = []
            for focal_idx, (idx, dist) in enumerate(zip(indices, distances_rad)):
                keep = idx != focal_idx
                neighbor_indices.append(idx[keep])
                neighbor_distances.append(dist[keep] * EARTH_RADIUS_KM)
            known_counts = np.zeros((n_months, n_condos), dtype="int16")
            for focal_idx, neighbor_idx in enumerate(neighbor_indices):
                if len(neighbor_idx):
                    known_counts[:, focal_idx] = known_prior[:, neighbor_idx].sum(axis=1)
            feature_arrays[f"nearby_{tag}_known_condos_lag1m"] = known_counts
            print(
                f"{label} {tag}: geographic median candidate neighbors={np.median([len(x) for x in neighbor_indices]):.0f}"
            )
            for window in WINDOWS_MONTHS:
                median_arr = np.full((n_months, n_condos), np.nan, dtype="float64")
                weighted_arr = np.full((n_months, n_condos), np.nan, dtype="float64")
                observation_arr = np.zeros((n_months, n_condos), dtype="int32")
                condo_arr = np.zeros((n_months, n_condos), dtype="int16")
                for focal_idx in range(n_condos):
                    stats = trailing_stats_for_focal(
                        price_matrix,
                        neighbor_indices[focal_idx],
                        neighbor_distances[focal_idx],
                        window,
                    )
                    (
                        median_arr[:, focal_idx],
                        weighted_arr[:, focal_idx],
                        observation_arr[:, focal_idx],
                        condo_arr[:, focal_idx],
                    ) = stats
                    if (focal_idx + 1) % 250 == 0 or focal_idx + 1 == n_condos:
                        print(
                            f"  {label} {tag} window={window}m: {focal_idx + 1}/{n_condos} condos"
                        )
                feature_arrays[f"nearby_{tag}_median_psf_{window}m_lag1m"] = median_arr
                feature_arrays[f"nearby_{tag}_distance_weighted_mean_psf_{window}m_lag1m"] = (
                    weighted_arr
                )
                feature_arrays[f"nearby_{tag}_observed_condo_months_{window}m_lag1m"] = (
                    observation_arr
                )
                feature_arrays[f"nearby_{tag}_observed_condos_{window}m_lag1m"] = condo_arr
            median_6m = feature_arrays[f"nearby_{tag}_median_psf_6m_lag1m"]
            lagged = np.full_like(median_6m, np.nan)
            lagged[MOMENTUM_MONTHS:, :] = median_6m[:-MOMENTUM_MONTHS, :]
            valid_growth = (median_6m > 0) & (lagged > 0)
            growth = np.full_like(median_6m, np.nan)
            growth[valid_growth] = np.log(median_6m[valid_growth] / lagged[valid_growth])
            feature_arrays[f"nearby_{tag}_median_psf_6m_log_growth_12m_lag1m"] = growth
        table_data = {
            "_month_key": np.repeat(months.to_numpy(), n_condos),
            "_condo_key": np.tile(np.asarray(condo_keys, dtype=object), n_months),
        }
        table_data.update(
            {name: values.reshape(-1).astype("float32") for name, values in feature_arrays.items()}
        )
        feature_table = pd.DataFrame(table_data)
        feature_cols = list(feature_arrays)
        first_month_mask = feature_table["_month_key"].eq(feature_table["_month_key"].min())
        lagged_count_cols = [
            col for col in feature_cols if "observed_condo_months" in col or "known_condos" in col
        ]
        if not (feature_table.loc[first_month_mask, lagged_count_cols] == 0).all().all():
            raise AssertionError(f"{label}: lag audit failed at first month")
        if feature_table.duplicated(["_month_key", "_condo_key"]).any():
            raise AssertionError(f"{label}: duplicate condo-month feature keys")
        if np.isinf(feature_table[feature_cols].to_numpy(dtype=float)).any():
            raise ValueError(f"{label}: infinite nearby feature values")
        metadata = {
            "rows": panel_rows,
            "condos": n_condos,
            "months": n_months,
            "observed_rows": observed_rows,
            "feature_cols": feature_cols,
        }
        del observations, condo_month, price_matrix, observed_matrix, known_prior, feature_arrays
        gc.collect()
        return (feature_table, metadata)

    print("[nearby_prices] phase 3")

    def save_compact_feature_table(feature_table, output_path):
        export = feature_table.rename(columns={"_month_key": "date", "_condo_key": "condo_key"})
        partial_path = output_path.parent / f"{output_path.name}.partial"
        export.to_csv(partial_path, index=False)
        partial_path.replace(output_path)
        print(f"Saved compact feature table: {output_path} ({len(export):,} rows)")

    def append_features_chunked(input_path, output_path, feature_table, feature_cols, label):
        input_cols = pd.read_csv(input_path, nrows=0).columns.tolist()
        required = {"project_idx", "condo_name", "date", "district"}
        missing = required - set(input_cols)
        if missing:
            raise KeyError(f"{label}: missing input columns {sorted(missing)}")
        overlap = sorted(set(feature_cols) & set(input_cols))
        if overlap:
            raise ValueError(f"{label}: nearby features already exist: {overlap}")
        if feature_table.duplicated(["_month_key", "_condo_key"]).any():
            raise AssertionError(f"{label}: duplicate feature-table keys")
        partial_path = output_path.parent / f"{output_path.name}.partial"
        rows_read = 0
        rows_written = 0
        feature_nonnull = pd.Series(0, index=feature_cols, dtype="int64")
        actual_districts = set()
        duplicate_keys = 0
        previous_last_key = None
        for chunk_number, chunk in enumerate(
            pd.read_csv(input_path, chunksize=WRITE_CHUNK_SIZE, low_memory=False), start=1
        ):
            before = len(chunk)
            month_key = (
                pd.to_datetime(chunk["date"], errors="coerce").dt.to_period("M").dt.start_time
            )
            condo_key = normalize_condo(chunk["condo_name"])
            if month_key.isna().any() or condo_key.isna().any():
                raise ValueError(f"{label}, chunk {chunk_number}: invalid date or condo name")
            district_numeric = pd.to_numeric(chunk["district"], errors="coerce")
            actual_districts.update(district_numeric.dropna().astype(int).unique())
            keys = pd.MultiIndex.from_arrays([chunk["project_idx"], month_key])
            if not keys.is_monotonic_increasing:
                raise AssertionError(
                    f"{label}: project_idx/date not sorted in chunk {chunk_number}"
                )
            duplicate_keys += int(keys.duplicated().sum())
            if len(keys) and previous_last_key == keys[0]:
                duplicate_keys += 1
            if len(keys):
                previous_last_key = keys[-1]
            helper = pd.DataFrame({"_month_key": month_key, "_condo_key": condo_key})
            chunk_with_keys = pd.concat(
                [chunk.reset_index(drop=True), helper.reset_index(drop=True)], axis=1
            )
            merged = chunk_with_keys.merge(
                feature_table,
                on=["_month_key", "_condo_key"],
                how="left",
                validate="many_to_one",
                sort=False,
            )
            if len(merged) != before:
                raise AssertionError(f"{label}: chunk {chunk_number} row count changed")
            if merged.shape[1] != len(input_cols) + 2 + len(feature_cols):
                raise AssertionError(f"{label}: unexpected merged column count")
            if np.isinf(merged[feature_cols].to_numpy(dtype=float)).any():
                raise ValueError(f"{label}: infinite merged nearby values")
            feature_nonnull += merged[feature_cols].notna().sum().astype("int64")
            merged = merged.drop(columns=["_month_key", "_condo_key"])
            merged.to_csv(
                partial_path,
                mode="w" if chunk_number == 1 else "a",
                header=chunk_number == 1,
                index=False,
            )
            rows_read += before
            rows_written += len(merged)
            if chunk_number == 1 or chunk_number % 5 == 0:
                print(f"  {label}: wrote {rows_written:,} rows")
            del chunk, helper, chunk_with_keys, merged, keys
            gc.collect()
        if rows_read != rows_written:
            raise AssertionError(f"{label}: row count changed {rows_read:,} -> {rows_written:,}")
        if duplicate_keys:
            raise AssertionError(f"{label}: {duplicate_keys:,} duplicate project-month keys")
        missing_districts = EXPECTED_DISTRICTS - actual_districts
        unexpected_districts = actual_districts - EXPECTED_DISTRICTS
        partial_path.replace(output_path)
        coverage = (100 * feature_nonnull / rows_written).rename("non_null_pct").reset_index()
        coverage = coverage.rename(columns={"index": "feature"})
        summary = {
            "dataset": label,
            "input_rows": rows_read,
            "output_rows": rows_written,
            "input_columns": len(input_cols),
            "features_added": len(feature_cols),
            "output_columns": len(input_cols) + len(feature_cols),
            "district_count": len(actual_districts),
            "missing_expected_districts": sorted(missing_districts),
            "unexpected_districts": sorted(unexpected_districts),
            "duplicate_project_months": duplicate_keys,
            "output_path": str(output_path),
        }
        print(
            f"{label}: rows {rows_read:,} -> {rows_written:,}; columns {len(input_cols)} -> {len(input_cols) + len(feature_cols)}"
        )
        print(
            f"{label}: districts={len(actual_districts)}, missing={sorted(missing_districts)}, unexpected={sorted(unexpected_districts)}"
        )
        print("Saved:", output_path)
        display(coverage)
        return (summary, coverage)

    print("[nearby_prices] phase 4")
    rent_features, rent_meta = build_nearby_feature_table(RENT_INPUT, "rent_psf_obs", "Rental")
    save_compact_feature_table(rent_features, RENT_FEATURE_OUTPUT)
    rental_summary, rental_coverage = append_features_chunked(
        RENT_INPUT, RENT_OUTPUT, rent_features, rent_meta["feature_cols"], "Rental"
    )
    display(rent_features[rent_meta["feature_cols"]].describe().T[["count", "mean", "min", "max"]])
    del rent_features
    gc.collect()
    print("[nearby_prices] phase 5")
    sales_features, sales_meta = build_nearby_feature_table(SALES_INPUT, "sale_psf_obs", "Sales")
    save_compact_feature_table(sales_features, SALES_FEATURE_OUTPUT)
    sales_summary, sales_coverage = append_features_chunked(
        SALES_INPUT, SALES_OUTPUT, sales_features, sales_meta["feature_cols"], "Sales"
    )
    display(
        sales_features[sales_meta["feature_cols"]].describe().T[["count", "mean", "min", "max"]]
    )
    del sales_features
    gc.collect()
    print("[nearby_prices] phase 6")
    summary = pd.DataFrame([rental_summary, sales_summary])
    display(summary)
    for label, path, meta in [
        ("Rental", RENT_OUTPUT, rent_meta),
        ("Sales", SALES_OUTPUT, sales_meta),
    ]:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        missing_features = sorted(set(meta["feature_cols"]) - set(columns))
        helper_columns = sorted(set(columns) & {"_month_key", "_condo_key"})
        duplicate_columns = pd.Index(columns)[pd.Index(columns).duplicated()].tolist()
        if missing_features or helper_columns or duplicate_columns:
            raise AssertionError(
                f"{label}: missing={missing_features}, helpers={helper_columns}, duplicates={duplicate_columns}"
            )
        print(path.name, "size (MB):", round(path.stat().st_size / 1024**2, 1))
    print("All full-district nearby-condo outputs created successfully.")
    return None
