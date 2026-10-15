#!/usr/bin/env python3
"""Check this release snapshot's SHA-256 inventory using only the standard library.

This does not validate training execution. See results/verify_results.py for
saved-number verification. Intentional edits require a new reviewed inventory.
"""
from pathlib import Path
import hashlib
import json
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "release_manifest.json").read_text())
    errors = []
    total = 0
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = ROOT / relative
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"Invalid inventory path: {relative}")
            continue
        if not path.is_file() or path.is_symlink():
            errors.append(f"Missing or non-regular file: {relative}")
            continue
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            errors.append(f"Changed file: {relative}")
        total += len(data)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: {len(manifest['files'])} inventoried files, {total:,} bytes.")
    print("Scope: packaged file integrity; no training or model evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
