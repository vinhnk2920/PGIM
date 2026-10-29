"""consulting. Explicit callable; no work is performed on import."""

from functions.common import display
from functions.features.panel_io import read_panel_chunks


def run(settings):
    print("[consulting] phase 1")
    import gc
    import re
    import numpy as np
    import pandas as pd

    CONSULTING_DIR = settings.prep_path("consulting")
    MODEL_INPUT_DIR = settings.folder("03_features")
    CONSULTING_CSV = CONSULTING_DIR / "consulting_prediction_manual_data.csv"
    RENTING_CSV = settings.folder("03_features") / "rental_labour.csv"
    SALES_CSV = settings.folder("03_features") / "sales_labour.csv"
    RENTING_OUT = settings.folder("03_features") / "rental_consulting.csv"
    SALES_OUT = settings.folder("03_features") / "sales_consulting.csv"
    CHUNK_SIZE = 100000
    EXPECTED_DISTRICTS = set(range(1, 29)) - {24}
    SENTIMENT_BACKEND = "finbert"
    required_files = [CONSULTING_CSV, RENTING_CSV, SALES_CSV]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing_files))
    MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Consulting data:", CONSULTING_CSV)
    print("Rental input:", RENTING_CSV)
    print("Sales input:", SALES_CSV)
    print("Rental output:", RENTING_OUT)
    print("Sales output:", SALES_OUT)
    print("[consulting] phase 2")
    df_con = pd.read_csv(CONSULTING_CSV)
    required_consulting_cols = {
        "forecast_for_year",
        "pub_date",
        "sale_price_forecast",
        "rental_price_forecast",
        "segment",
    }
    missing = required_consulting_cols - set(df_con.columns)
    if missing:
        raise KeyError(f"Consulting file is missing columns: {sorted(missing)}")
    df_con["forecast_for_year"] = pd.to_numeric(
        df_con["forecast_for_year"], errors="coerce"
    ).astype("Int64")
    MONTH_MAP = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }

    def parse_pub_period(value):
        try:
            text = str(value).strip()
            if re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", text):
                dt = pd.to_datetime(text, errors="raise")
                return dt.year * 100 + dt.month
            yy_text, month_text = str(value).strip().split("-")
            yy = int(yy_text)
            year = 2000 + yy if yy < 100 else yy
            return year * 100 + MONTH_MAP[month_text.title()]
        except Exception:
            return np.nan

    df_con["pub_period"] = df_con["pub_date"].apply(parse_pub_period).astype("Int64")
    print(f"Consulting rows: {len(df_con):,}")
    print(f"Invalid publication months: {df_con['pub_period'].isna().sum():,}")
    display(df_con[["pub_date", "pub_period", "forecast_for_year"]].head())
    print("[consulting] phase 3")

    def parse_midpoint(text):
        if pd.isna(text) or not isinstance(text, str):
            return np.nan
        numbers = re.findall("[-+]?\\d+(?:\\.\\d+)?", text.strip())
        if not numbers:
            return np.nan
        values = [float(value) for value in numbers[:2]]
        return (values[0] if len(values) == 1 else np.mean(values)) / 100.0

    QUALITATIVE_DIRECTION_OVERRIDES = {
        "grow marginally": 1,
        "continue to fall": -1,
        "rise or hold steady": 1,
        "demand expected to remain stable": 0,
        "holding firm": 0,
        "hold up well": 0,
        "no change": 0,
        "flat": 0,
        "limited drop": -1,
        "in line with inflation": 1,
        "weigh down rents": -1,
        "price increases": 1,
        "lose top spot in sales volume to mid-tier": np.nan,
        "become market leader in sales volume": np.nan,
        "sales volume expected to improve": np.nan,
    }
    _finbert_tokenizer = None
    _finbert_model = None

    def load_finbert():
        nonlocal _finbert_tokenizer, _finbert_model
        if _finbert_model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            _finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            _finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            _finbert_model.eval()

    def finbert_direction(text):
        import torch

        load_finbert()
        inputs = _finbert_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = _finbert_model(**inputs).logits
        label_index = int(torch.argmax(logits, dim=-1))
        label = _finbert_model.config.id2label[label_index].lower()
        return {"positive": 1, "negative": -1}.get(label, 0)

    def parse_direction(text):
        if pd.isna(text) or not isinstance(text, str):
            return np.nan
        midpoint = parse_midpoint(text)
        if pd.notna(midpoint):
            if midpoint > 0.005:
                return 1
            if midpoint < -0.005:
                return -1
            return 0
        normalized = " ".join(text.strip().lower().split())
        if normalized in QUALITATIVE_DIRECTION_OVERRIDES:
            return QUALITATIVE_DIRECTION_OVERRIDES[normalized]
        if SENTIMENT_BACKEND == "finbert":
            return finbert_direction(text)
        raise ValueError(f"Unsupported sentiment backend: {SENTIMENT_BACKEND}")

    qualitative_values = pd.concat(
        [
            df_con.loc[df_con["rental_price_forecast"].notna(), "rental_price_forecast"],
            df_con.loc[df_con["sale_price_forecast"].notna(), "sale_price_forecast"],
        ]
    ).drop_duplicates()
    qualitative_values = qualitative_values[qualitative_values.apply(parse_midpoint).isna()]
    qualitative_audit = pd.DataFrame({"text": qualitative_values})
    qualitative_audit["direction"] = qualitative_audit["text"].apply(parse_direction)
    display(qualitative_audit)
    print("[consulting] phase 4")
    EXCLUDE_SEGMENTS = {"super luxury", "savills basket"}

    def build_agg_table(data, value_col):
        working = data.copy()
        working["_mid"] = working[value_col].apply(parse_midpoint)
        working["_dir"] = working[value_col].apply(parse_direction)
        working["_segment_norm"] = working["segment"].fillna("").str.strip().str.lower()
        valid = working[
            working[value_col].notna()
            & working["pub_period"].notna()
            & working["forecast_for_year"].notna()
        ].copy()
        records = []
        for (pub_period, forecast_year), group in valid.groupby(
            ["pub_period", "forecast_for_year"]
        ):
            overall = group[group["_segment_norm"] == "overall"]
            if len(overall):
                chosen = overall.iloc[-1]
                midpoint, direction, source_segment = (chosen["_mid"], chosen["_dir"], "Overall")
            else:
                fallback = group[~group["_segment_norm"].isin(EXCLUDE_SEGMENTS)]
                mids = fallback["_mid"].dropna()
                directions = fallback["_dir"].dropna()
                midpoint = float(np.median(mids)) if len(mids) else np.nan
                if pd.notna(midpoint):
                    direction = 1 if midpoint > 0.005 else -1 if midpoint < -0.005 else 0
                elif len(directions):
                    direction = int(directions.mode().iloc[0])
                else:
                    direction = np.nan
                source_segment = f"median({len(fallback)} segments)"
            records.append(
                {
                    "pub_period": int(pub_period),
                    "forecast_for_year": int(forecast_year),
                    "consulting_forecast_mid": midpoint,
                    "consulting_direction": direction,
                    "source_segment": source_segment,
                }
            )
        if not records:
            raise ValueError(
                f"No usable consulting forecasts in {value_col}; check publication dates and forecast years"
            )
        result = pd.DataFrame(records).sort_values(["forecast_for_year", "pub_period"])
        if result.duplicated(["pub_period", "forecast_for_year"]).any():
            raise AssertionError("Aggregated consulting table contains duplicate keys")
        return result

    df_agg_rental = build_agg_table(df_con, "rental_price_forecast")
    df_agg_sale = build_agg_table(df_con, "sale_price_forecast")
    print("Aggregated rental rows:", len(df_agg_rental))
    print("Aggregated sale rows:", len(df_agg_sale))
    print("[consulting] phase 5")

    def lookup_consulting(origin_period, aggregated):
        origin_year = int(origin_period) // 100
        candidates = aggregated[
            (aggregated["forecast_for_year"] == origin_year)
            & (aggregated["pub_period"] <= int(origin_period))
        ]
        if candidates.empty:
            return (np.nan, np.nan, np.nan, None)
        chosen = candidates.loc[candidates["pub_period"].idxmax()]
        return (
            chosen["consulting_forecast_mid"],
            chosen["consulting_direction"],
            int(chosen["pub_period"]),
            chosen["source_segment"],
        )

    def build_monthly_lookup(aggregated, prefix, start="1995-01", end="2026-12"):
        months = pd.period_range(start, end, freq="M")
        records = []
        for month in months:
            period = month.year * 100 + month.month
            midpoint, direction, source_pub, source_segment = lookup_consulting(period, aggregated)
            records.append(
                {
                    "period": period,
                    f"consulting_{prefix}_forecast_mid": midpoint,
                    f"consulting_{prefix}_direction": direction,
                    "source_pub_period": source_pub,
                    "source_segment": source_segment,
                }
            )
        lookup = pd.DataFrame(records)
        leaked = lookup[
            lookup["source_pub_period"].notna() & (lookup["source_pub_period"] > lookup["period"])
        ]
        if len(leaked):
            raise AssertionError(f"{prefix}: future-report leakage found in {len(leaked)} months")
        return lookup

    lookup_rental = build_monthly_lookup(df_agg_rental, "rental")
    lookup_sale = build_monthly_lookup(df_agg_sale, "sale")
    print(
        "Future-report leakage — rental:",
        int((lookup_rental["source_pub_period"] > lookup_rental["period"]).sum()),
    )
    print(
        "Future-report leakage — sale:",
        int((lookup_sale["source_pub_period"] > lookup_sale["period"]).sum()),
    )
    display(lookup_rental.loc[lookup_rental["period"].isin([200801, 200805, 200901, 202001])])
    display(lookup_sale.loc[lookup_sale["period"].isin([200801, 200804, 200806, 200810, 202001])])
    print("[consulting] phase 6")

    def merge_consulting_file(
        input_path, output_path, label, lookup, feature_cols, chunk_size=CHUNK_SIZE
    ):
        print(f"\n--- {label} ---")
        partial_path = output_path.parent / f"{output_path.name}.partial"
        total_input_rows = 0
        total_output_rows = 0
        base_columns = None
        actual_districts = set()
        duplicate_keys = 0
        previous_last_key = None
        non_null_counts = {col: 0 for col in feature_cols}
        first_valid = {col: pd.NaT for col in feature_cols}
        last_valid = {col: pd.NaT for col in feature_cols}
        lookup_for_merge = lookup[["period"] + feature_cols]
        if lookup_for_merge["period"].duplicated().any():
            raise AssertionError(f"{label}: duplicate lookup periods")
        for chunk_no, frame in enumerate(
            read_panel_chunks(input_path, chunk_size, consolidate=True), start=1
        ):
            if chunk_no == 1:
                base_columns = len(frame.columns)
                required = {"project_idx", "date", "district"}
                missing = required - set(frame.columns)
                if missing:
                    raise KeyError(f"{label}: missing columns {sorted(missing)}")
                overlap = sorted(set(feature_cols) & set(frame.columns))
                if overlap:
                    raise ValueError(f"{label}: consulting columns already exist: {overlap}")
            rows_this_chunk = len(frame)
            total_input_rows += rows_this_chunk
            frame["date"] = (
                pd.to_datetime(frame["date"], errors="coerce").dt.to_period("M").dt.start_time
            )
            if frame["date"].isna().any():
                raise ValueError(f"{label}, chunk {chunk_no}: invalid dates found")
            frame["period"] = frame["date"].dt.year * 100 + frame["date"].dt.month
            district_numeric = pd.to_numeric(frame["district"], errors="coerce")
            actual_districts.update(district_numeric.dropna().astype(int).unique())
            keys = pd.MultiIndex.from_frame(frame[["project_idx", "date"]])
            if not keys.is_monotonic_increasing:
                raise AssertionError(f"{label}: project_idx/date is not sorted in chunk {chunk_no}")
            duplicate_keys += int(keys.duplicated().sum())
            if len(keys) and previous_last_key == keys[0]:
                duplicate_keys += 1
            if len(keys):
                previous_last_key = keys[-1]
            merged = frame.merge(lookup_for_merge, on="period", how="left", validate="many_to_one")
            if len(merged) != rows_this_chunk:
                raise AssertionError(f"{label}, chunk {chunk_no}: row count changed")
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
            merged = merged.drop(columns=["period"])
            merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
            merged.to_csv(
                partial_path, mode="w" if chunk_no == 1 else "a", header=chunk_no == 1, index=False
            )
            total_output_rows += len(merged)
            if chunk_no == 1 or chunk_no % 5 == 0:
                print(f"  processed {total_output_rows:,} rows")
            del frame, merged, keys
            gc.collect()
        if total_input_rows != total_output_rows:
            raise AssertionError(f"{label}: total row count changed")
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
            "input_columns": base_columns,
            "output_columns": base_columns + len(feature_cols),
            "district_count": len(actual_districts),
            "missing_expected_districts": sorted(missing_districts),
            "unexpected_districts": sorted(unexpected_districts),
            "duplicate_project_months": duplicate_keys,
            "output_path": str(output_path),
        }
        print(f"Rows: {total_input_rows:,} -> {total_output_rows:,} (unchanged)")
        print(
            f"Districts: {len(actual_districts)}; missing: {sorted(missing_districts)}; unexpected: {sorted(unexpected_districts)}"
        )
        print("Saved:", output_path)
        display(coverage)
        return (summary, coverage)

    RENTAL_FEATURES = ["consulting_rental_forecast_mid", "consulting_rental_direction"]
    SALE_FEATURES = ["consulting_sale_forecast_mid", "consulting_sale_direction"]
    rental_summary, rental_coverage = merge_consulting_file(
        RENTING_CSV, RENTING_OUT, "Rental", lookup_rental, RENTAL_FEATURES
    )
    sales_summary, sales_coverage = merge_consulting_file(
        SALES_CSV, SALES_OUT, "Sales", lookup_sale, SALE_FEATURES
    )
    print("[consulting] phase 7")
    summary = pd.DataFrame([rental_summary, sales_summary])
    display(summary)
    for path in [RENTING_OUT, SALES_OUT]:
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
