"""Merge authoritative lookup fields without retaining stale suffixed copies."""


def merge_reference(frame, reference, *, on=None, left_on=None, right_on=None):
    """Refresh reference-owned columns while preserving left rows and other fields."""
    left_keys = [on or left_on] if isinstance(on or left_on, str) else list(on or left_on)
    protected = set(left_keys)
    refresh = [column for column in reference.columns if column in frame.columns and column not in protected]
    result = frame.drop(columns=refresh).merge(
        reference, on=on, left_on=left_on, right_on=right_on, how="left", validate="many_to_one"
    )
    if len(result) != len(frame):
        raise ValueError("Reference merge changed the transaction row count")
    return result
