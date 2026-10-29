"""A 99.co snapshot can already contain coordinates and other lookup fields."""

import pandas as pd
import pytest

from functions.preprocessing.merging import merge_reference


def test_reference_replaces_stale_enrichment_without_suffixes_or_row_changes():
    frame = pd.DataFrame({"Project Name": ["A", "A", "B"], "project_name": ["STALE"] * 3, "latitude": [9, 9, 8],
                          "longitude": [7, 7, 6], "tenure": [99, 99, 999], "price": [10, 20, 30]})
    lookup = pd.DataFrame({"project_name": ["A"], "latitude": [1.3], "longitude": [103.8]})
    result = merge_reference(frame, lookup, left_on="Project Name", right_on="project_name")
    assert len(result) == 3
    assert result["latitude"].tolist()[:2] == [1.3, 1.3]
    assert result.loc[2, ["latitude", "longitude"]].isna().all()
    assert not any(name.endswith(("_x", "_y")) for name in result)
    pd.testing.assert_series_equal(result["tenure"], frame["tenure"])
    pd.testing.assert_series_equal(result["price"], frame["price"])
    pd.testing.assert_series_equal(frame["latitude"], pd.Series([9, 9, 8], name="latitude"))


def test_duplicate_reference_keys_fail_instead_of_multiplying_transactions():
    with pytest.raises(pd.errors.MergeError):
        merge_reference(pd.DataFrame({"project": ["A"]}),
                        pd.DataFrame({"project": ["A", "A"], "latitude": [1.3, 1.4]}), on="project")
