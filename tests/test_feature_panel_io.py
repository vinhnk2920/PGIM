"""Prevent wide-panel fragmentation and mixed neighbourhood inference warnings."""

import warnings

import numpy as np
import pandas as pd

from functions.features.panel_io import read_panel_chunks


def test_neighbourhood_type_is_stable_across_empty_and_text_chunks(tmp_path):
    path = tmp_path / "panel.csv"
    path.write_text("Neighbourhood,date,value\n,2020-01,1\n,2020-02,2\nCentral,2020-03,3\nEast,2020-04,4\n", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.DtypeWarning)
        chunks = list(read_panel_chunks(path, 2))
    assert len(chunks) == 2
    assert all(str(frame["Neighbourhood"].dtype) == "string" for frame in chunks)
    assert chunks[0]["Neighbourhood"].isna().all()
    assert chunks[1]["Neighbourhood"].tolist() == ["Central", "East"]
    assert pd.concat(chunks)["value"].tolist() == [1, 2, 3, 4]


def test_wide_chunk_can_add_helper_columns_without_fragmentation_warning(monkeypatch):
    from functions.features import panel_io

    # Public concat creates separate numeric blocks, reproducing a wide input
    # frame without relying on pandas' private block-manager API.
    frame = pd.concat([pd.DataFrame({f"value_{index}": np.arange(4, dtype=float)})
                       for index in range(120)], axis=1)
    original = frame.copy()

    class Reader:
        def __enter__(self):
            return iter([frame])

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(panel_io.pd, "read_csv", lambda *args, **kwargs: Reader())
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        chunk = next(read_panel_chunks("unused.csv", 4, consolidate=True))
        chunk["period"] = [202001, 202002, 202003, 202004]
    pd.testing.assert_frame_equal(chunk.drop(columns="period"), original, check_exact=True)
    assert "period" not in frame
