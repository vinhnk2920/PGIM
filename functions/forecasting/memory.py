"""Memory reduction shared by sales and rental feature preparation."""

import numpy as np
import pandas as pd


def downscale_numeric_frame(frame):
    before = frame.memory_usage(deep=True).sum()
    for col in frame.select_dtypes(include=["float64"]):
        frame[col] = pd.to_numeric(frame[col], downcast="float")
    for col in frame.select_dtypes(include=["int64"]):
        values = frame[col]
        if (
            values.notna().all()
            and values.min() >= np.iinfo(np.int32).min
            and values.max() <= np.iinfo(np.int32).max
        ):
            frame[col] = values.astype("int32")
    after = frame.memory_usage(deep=True).sum()
    print(f"Numeric downscale: {before / 2**30:.2f} -> {after / 2**30:.2f} GiB")
    return frame
