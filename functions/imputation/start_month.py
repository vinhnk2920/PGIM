"""Restrict production imputation to the part of each condo's history in scope."""

from collections import defaultdict
from datetime import datetime
import re

import numpy as np
import pandas as pd


def _project_key(name):
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", name.upper()).split())


def _read_top_months(path):
    if not path.is_file():
        print(f"TOP date source unavailable: {path}. Using first observed transactions.")
        return {}

    columns = ["Project Name", "Project Name in Realis", "TOP date"]
    details = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
    months = defaultdict(set)
    years = defaultdict(set)
    for name, alias, value in details[columns].itertuples(index=False, name=None):
        keys = {_project_key(name), _project_key(alias)} - {""}
        value = value.strip()
        year_only = re.fullmatch(r"((?:19|20)\d{2})(?:\s*\(\d+ years?\))?", value, re.I)
        if year_only:
            for key in keys:
                years[key].add(int(year_only[1]))
            continue

        match = re.fullmatch(r"\d{1,2}-[A-Za-z]{3,9}-(\d{2}|\d{4})", value)
        if not match:
            continue
        date_format = "%d-%b-%y" if len(match[1]) == 2 else "%d-%b-%Y"
        try:
            date = datetime.strptime(re.sub(r"Sept", "Sep", value, flags=re.I), date_format)
        except ValueError:
            continue
        for key in keys:
            if (date.month, date.day) == (1, 1):
                # The source commonly uses 1 January for year-level records.
                years[key].add(date.year)
            else:
                months[key].add(date.year * 100 + date.month)

    lookup = {}
    for key, dates in months.items():
        if len(dates) == 1:
            [month] = dates
            if not years[key] or years[key] == {month // 100}:
                lookup[key] = month
    return lookup


def apply_top_first(top_path, series_meta, all_months, observation_mask, *outputs):
    """Blank pre-start unobserved cells in the exported arrays, preserving the grid."""
    observed = observation_mask.astype(bool)
    by_series_month = observed.any(axis=2)
    names = series_meta.set_index("project_idx")["condo_name"].reindex(range(len(observed)))
    if names.isna().any() or not by_series_month.any(axis=1).all():
        raise ValueError("TOP-first requires a condo name and observed month for every series")
    if any(array.shape != observed.shape for array in outputs):
        raise ValueError("Production arrays must match the imputation panel shape")

    months = np.array([int(month.replace("-", "")) for month in all_months])
    first = pd.Series(months[by_series_month.argmax(axis=1)], index=names.to_numpy())
    first_by_condo = first.groupby(level=0, sort=False).min()
    lookup = _read_top_months(top_path)
    top = first_by_condo.index.to_series().map(lambda name: lookup.get(_project_key(name)))
    start_by_condo = top.fillna(first_by_condo)
    before_start = months[None, :] < names.map(start_by_condo).to_numpy()[:, None]
    exclude = before_start[:, :, None] & ~observed
    for array in outputs:
        array[exclude] = np.nan

    n_top = int(top.notna().sum())
    excluded_count = int(exclude.sum())
    print(
        f"TOP-first: {n_top:,} condos with a TOP month; "
        f"{len(top) - n_top:,} using first observed transaction; "
        f"{excluded_count:,} pre-start missing cells left blank."
    )
    return excluded_count
