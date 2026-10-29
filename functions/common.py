"""Shared calendar arithmetic, console reporting and audit logs."""

import json
import sys
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter


def add_months(period, months):
    year, month = divmod(int(period), 100)
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid YYYYMM period: {period}")
    ordinal = year * 12 + month - 1 + months
    return (ordinal // 12) * 100 + ordinal % 12 + 1


def rolling_windows(start, end, train_months=180, test_months=12):
    """Nonoverlapping forecast origins and a fixed-width calendar training pool."""
    if train_months < 1 or test_months < 1 or start > end:
        raise ValueError("Invalid rolling window range or width")
    windows = []
    while start <= end:
        windows.append(
            dict(
                train_start=add_months(start, -train_months),
                train_end=add_months(start, -1),
                test_start=start,
                test_end=min(end, add_months(start, test_months - 1)),
            )
        )
        start = add_months(start, test_months)
    return windows


def display(*items):
    """Plain terminal replacement for notebook display()."""
    for item in items:
        print(item.to_string() if hasattr(item, "to_string") else item)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


@contextmanager
def run_log(settings, label):
    """Print progress and keep the exact console transcript and completion status."""
    from contextlib import redirect_stdout, redirect_stderr

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = settings.folder("logs")
    status = dict(stage=label, started=stamp, status="running", settings=vars(settings))
    start = perf_counter()
    with (folder / f"{stamp}_{label}.log").open("w", encoding="utf-8") as handle:
        with redirect_stdout(Tee(sys.stdout, handle)), redirect_stderr(Tee(sys.stderr, handle)):
            print(f"[START] {label}")
            try:
                yield
            except BaseException as exc:
                status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"[FAILED] {label}: {exc}")
                raise
            else:
                status["status"] = "completed"
                print(f"[DONE] {label}")
            finally:
                status["elapsed_seconds"] = round(perf_counter() - start, 3)
                (folder / f"{stamp}_{label}.json").write_text(
                    json.dumps(status, default=str, indent=2), encoding="utf-8"
                )
