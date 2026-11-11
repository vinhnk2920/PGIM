"""Read project/date reference attributes without retaining redundant transactions."""

import pandas as pd


def rental_property_reference(path, raw_columns):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    keys = ["Project Name"]
    if "Lease Commencement Date" in header and "Lease Commencement Date" in raw_columns:
        keys.append("Lease Commencement Date")
    extra = [c for c in header if c not in raw_columns and c not in keys]
    pieces = []
    for chunk in pd.read_csv(path, usecols=keys + extra, low_memory=False, chunksize=100000):
        chunk["Project Name"] = chunk["Project Name"].astype(str).str.strip().str.upper()
        if "Lease Commencement Date" in keys:
            chunk["Lease Commencement Date"] = pd.to_datetime(chunk["Lease Commencement Date"], errors="coerce")
        pieces.append(chunk.drop_duplicates(keys, keep="first"))
    return pd.concat(pieces, ignore_index=True).drop_duplicates(keys, keep="first")
