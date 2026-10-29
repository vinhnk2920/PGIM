"""Build and load the two supply reference tables without changing feature rules."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone, timedelta

import pandas as pd

SUPPLY_FILES = {
    "developer_sales": "developer_sales.csv",
    "project_details": "project_details.csv",
}
PROVENANCE_COLUMNS = ["source_file", "source_row"]


def read_original_supply(directory, kind):
    """Use the legacy per-file parsing and filename order; never skip a failed file."""
    if kind not in SUPPLY_FILES:
        raise ValueError(f"Unknown supply table: {kind}")
    files = sorted(Path(directory).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No original supply CSVs found in {directory}")
    frames, records = [], []
    pattern = r"D(\d{2})_(\d{4})_(\d{2})" if kind == "developer_sales" else r"D(\d{2})_(\d{4})_Q(\d)"
    for path in files:
        match = re.search(pattern, path.name)
        if not match:
            raise ValueError(f"Cannot extract supply district and period from {path.name}")
        district, year, period = map(int, match.groups())
        if district not in range(1, 29) or period not in range(1, 13 if kind == "developer_sales" else 5):
            raise ValueError(f"Invalid supply district or period in {path.name}")
        options = {"thousands": ","} if kind == "developer_sales" else {}
        try:
            frame = pd.read_csv(path, **options)
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="latin-1", **options)
        if set(PROVENANCE_COLUMNS) & set(frame.columns):
            raise ValueError(f"Reserved provenance column in {path.name}")
        frame["district"] = district
        if kind == "project_details":
            frame["year"], frame["quarter"] = year, period
        frame["source_file"] = path.name
        frame["source_row"] = range(1, len(frame) + 1)
        frames.append(frame)
        records.append({"file": path.name, "rows": len(frame), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    combined = pd.concat(frames, ignore_index=True)
    print(f"[supply consolidation] {kind}: {len(files):,} files, {len(combined):,} rows")
    return combined, records


def load_supply_table(path, kind):
    """Preserve parsed text/markers, then restore integer filename metadata."""
    if kind not in SUPPLY_FILES:
        raise ValueError(f"Unknown supply table: {kind}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Consolidated supply input not found: {path}. "
            "Run python consolidate_supply.py to rebuild it from the external original downloads; "
            "use --developer-sales-dir and --project-details-dir if their locations have changed."
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    metadata = ["district", "source_row"]
    required = {"Project Name", "source_file", *metadata}
    if kind == "developer_sales":
        required |= {"Reference Month", "Total Number of Units in Project", "Cumulative Units Sold to-date", "Units Launched in the Month"}
    else:
        metadata += ["year", "quarter"]
        required |= {"year", "quarter"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Supply input {path.name} is missing columns: {sorted(missing)}")
    for column in metadata:
        numeric = pd.to_numeric(frame[column], errors="raise")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"Supply metadata {column} must contain integers: {path}")
        frame[column] = numeric.astype("int64")
    if frame.empty or not frame["district"].between(1, 28).all():
        raise ValueError(f"Empty supply input or invalid districts: {path}")
    if not frame["source_row"].ge(1).all() or frame["source_file"].isna().any():
        raise ValueError(f"Invalid supply provenance: {path}")
    if frame.duplicated(PROVENANCE_COLUMNS).any():
        raise ValueError(f"Repeated source file/row identifiers: {path}")
    if kind == "project_details" and not frame["quarter"].between(1, 4).all():
        raise ValueError(f"Invalid supply quarters: {path}")
    return frame


def consolidate_supply(supply_root, report_dir, *, developer_sales_dir, project_details_dir):
    """Rebuild from explicit external source directories without copying their files."""
    supply_root, report_dir = Path(supply_root), Path(report_dir)
    supply_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    source_dirs = {"developer_sales": Path(developer_sales_dir),
                   "project_details": Path(project_details_dir)}
    pending, reports = [], {}
    try:
        for kind, filename in SUPPLY_FILES.items():
            frame, sources = read_original_supply(source_dirs[kind], kind)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{kind}_", suffix=".csv", dir=supply_root)
            os.close(descriptor)
            temporary = Path(temporary_name)
            target = supply_root / filename
            pending.append((temporary, target))
            frame.to_csv(temporary, index=False)
            restored = load_supply_table(temporary, kind)
            # CSV stores scalar text rather than mixed Python scalar types. Compare
            # every parsed value as text, with the same missing-value representation.
            pd.testing.assert_frame_equal(frame.astype("string"), restored.astype("string"), check_exact=True)
            reports[kind] = {"output": filename, "rows": len(frame), "columns": len(frame.columns),
                             "source_directory": str(source_dirs[kind].resolve()),
                             "source_files": len(sources), "sources": sources,
                             "bytes": temporary.stat().st_size,
                             "sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
                             "parsed_values_and_order_preserved": True}
        for temporary, target in pending:
            temporary.replace(target)
        report = {"created_at_singapore": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  "tables": reports, "originals_preserved": True,
                  "row_order": "Sorted source filenames, then original row order; no deduplication during consolidation.",
                  "parsing": "Legacy per-file parsing, including developer-sales thousands separators and Latin-1 fallback. Empty parsed values remain missing. Feature calculations retain their existing numeric conversion and deduplication rules."}
        (report_dir / "CONSOLIDATION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        for temporary, _ in pending:
            if temporary.exists():
                temporary.unlink()
