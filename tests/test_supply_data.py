"""Regression checks for supply file ordering, metadata and numeric parsing."""

import pandas as pd
import pytest

from functions.features.supply_data import consolidate_supply, load_supply_table


def test_consolidation_preserves_rows_markers_and_filename_metadata(tmp_path):
    root = tmp_path / "supply"
    dev = tmp_path / "external" / "developer_sales"
    proj = tmp_path / "external" / "project_details"
    dev.mkdir(parents=True)
    proj.mkdir(parents=True)
    # Create in reverse order; duplicate project/month records must retain the
    # same source precedence as the legacy sorted-file concatenation.
    names = ["dev_sales_D02_2020_02_page_2.csv", "dev_sales_D02_2020_02_page_1.csv"]
    for name, units in zip(names, [20, 10]):
        pd.DataFrame({"Project Name": ["A"], "Reference Month": ["Feb 2020"],
                      "Total Number of Units in Project": ["1,234"],
                      "Cumulative Units Sold to-date": [units],
                      "Units Launched in the Month": [units],
                      "Median Price ($ PSF) in the Month": ["-"]}).to_csv(dev / name, index=False)
    pd.DataFrame({"Project Name": ["A", "B"],
                  "Total Units in Project by Construction Status Planned": ["1,234", "-"]}).to_csv(
                      proj / "project_details_D02_2020_Q1_page_1.csv", index=False)
    source_bytes = {p: p.read_bytes() for p in (tmp_path / "external").rglob("*.csv")}
    report = consolidate_supply(root, tmp_path / "report", developer_sales_dir=dev,
                                project_details_dir=proj)
    assert not (root / "originals").exists()
    developer = load_supply_table(root / "developer_sales.csv", "developer_sales")
    assert developer["source_file"].tolist() == sorted(names)
    assert developer["Cumulative Units Sold to-date"].tolist() == ["10", "20"]
    assert developer["Total Number of Units in Project"].tolist() == ["1234", "1234"]
    assert developer["Median Price ($ PSF) in the Month"].tolist() == ["-", "-"]
    details = load_supply_table(root / "project_details.csv", "project_details")
    assert details["district"].tolist() == [2, 2]
    assert details["year"].tolist() == [2020, 2020]
    assert details["quarter"].tolist() == [1, 1]
    assert details["Total Units in Project by Construction Status Planned"].tolist() == ["1,234", "-"]
    assert report["tables"]["developer_sales"]["source_files"] == 2
    assert all(path.read_bytes() == content for path, content in source_bytes.items())


def test_missing_summary_does_not_silently_scan_originals(tmp_path):
    with pytest.raises(FileNotFoundError, match="consolidate_supply.py"):
        load_supply_table(tmp_path / "developer_sales.csv", "developer_sales")


def test_bad_filename_fails_instead_of_creating_partial_summaries(tmp_path):
    from functions.features.supply_data import read_original_supply

    (tmp_path / "unknown.csv").write_text("Project Name\nA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot extract"):
        read_original_supply(tmp_path, "developer_sales")
