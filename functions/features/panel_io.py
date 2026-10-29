"""Shared chunk reader for feature merges on wide project/month panels."""

import pandas as pd


def read_panel_chunks(path, chunk_size, *, consolidate=False):
    """Keep neighbourhood labels textual; optionally consolidate numeric blocks."""
    with pd.read_csv(path, chunksize=chunk_size, low_memory=False,
                     dtype={"Neighbourhood": "string"}) as reader:
        for frame in reader:
            # A single copy consolidates a wide frame before callers insert
            # helper columns. Work stays bounded by the existing chunk size.
            yield frame.copy() if consolidate else frame
