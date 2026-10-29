"""Rebuild the two supply reference CSVs after adding original downloads."""

import argparse
from pathlib import Path

from functions.config import DEFAULT_SOURCE, PACKAGE_ROOT
from functions.features.supply_data import consolidate_supply

DEFAULT_ORIGINALS = PACKAGE_ROOT.parent / "data_prep" / "data_v2_supply_side"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE,
                        help="Destination data root for data_prep/supply/ consolidated CSVs")
    parser.add_argument("--developer-sales-dir", type=Path,
                        default=DEFAULT_ORIGINALS / "realis_developer_sale_downloads",
                        help="External directory containing original monthly developer-sales CSVs")
    parser.add_argument("--project-details-dir", type=Path,
                        default=DEFAULT_ORIGINALS / "project_project_details_end_of_quarter",
                        help="External directory containing original quarterly project-details CSVs")
    parser.add_argument("--report-dir", type=Path, default=PACKAGE_ROOT / "validation" / "supply",
                        help="Directory for source checksums and consolidation details")
    args = parser.parse_args()
    report = consolidate_supply(args.source_root / "data_prep" / "supply", args.report_dir,
                                developer_sales_dir=args.developer_sales_dir,
                                project_details_dir=args.project_details_dir)
    for kind, table in report["tables"].items():
        print(f"[saved] {kind}: {table['output']} ({table['rows']:,} rows)")
    print("Original downloads preserved. Normal feature runs read only the two consolidated CSVs.")


if __name__ == "__main__":
    main()
