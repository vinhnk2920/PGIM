"""Fail a fresh experiment if it reads an old price, feature or model checkpoint."""

import atexit
import hashlib
import json
import os
from pathlib import Path
import sys


class FreshInputPolicy:
    def __init__(self, source_root, output_root):
        self.source = Path(source_root).resolve()
        self.output = Path(output_root).resolve()
        self.project = self.source.parent
        self.seen = set()
        self.violations = []

    def check(self, path):
        target = Path(path).resolve()
        if target.is_relative_to(self.output):
            return
        forbidden = [self.source / name for name in ("imputation", "model_input", "snapshots")]
        forbidden += [self.project / "outputs", self.project.parent / "rolling_window_models",
                      self.project.parent / "imputation_v2"]
        if any(target.is_relative_to(root) for root in forbidden):
            self.violations.append(str(target))
            raise PermissionError(f"Fresh raw experiment attempted to read an archived checkpoint: {target}")
        if target.is_relative_to(self.source):
            self.seen.add(str(target))

    def hook(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode = args[1]
        if isinstance(mode, str) and any(char in mode for char in "wax+"):
            return
        self.check(args[0])

    def save(self):
        folder = self.output / "audit"
        folder.mkdir(parents=True, exist_ok=True)
        files = []
        for name in sorted(self.seen):
            path = Path(name)
            digest = None
            if path.is_file():
                checksum = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        checksum.update(block)
                digest = checksum.hexdigest()
            files.append({"path": name, "bytes": path.stat().st_size if path.is_file() else None,
                          "sha256": digest})
        (folder / "fresh_input_reads.json").write_text(json.dumps({
            "policy": "No old imputation, feature or forecasting checkpoint reads",
            "source_root": str(self.source), "output_root": str(self.output),
            "forbidden_reads": self.violations, "source_files_read": files,
        }, indent=2), encoding="utf-8")


def enable(settings):
    policy = FreshInputPolicy(settings.source_root, settings.output_root)
    sys.addaudithook(policy.hook)
    atexit.register(policy.save)
    return policy
