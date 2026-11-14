"""Immutable local file/array baselines, independent of the inference implementation.

record paths.json baseline.json -- paths.json maps labels to existing file paths.
check baseline.json            -- verify original paths have not changed.
compare baseline.json paths.json -- compare replacement outputs against baseline.

Any byte difference fails, even if numeric arrays happen to agree. Never rewrites
a baseline during checking. NPZ members include independent content digests to
distinguish serialization changes from numerical changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(path):
    path = Path(path).resolve(strict=True)
    value = {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}
    if path.suffix.lower() in (".npz", ".npy"):
        loaded = np.load(path, allow_pickle=False)
        try:
            members = {key: loaded[key] for key in loaded.files} if hasattr(loaded, "files") else {"array": loaded}
            value["arrays"] = {
                key: {"shape": list(a.shape), "dtype": a.dtype.str,
                      "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}
                for key, a in sorted(members.items())
            }
        finally:
            if hasattr(loaded, "close"):
                loaded.close()
    return value


def compare(baseline, paths):
    rows = {}
    for name, expected in baseline["files"].items():
        try:
            actual = describe(paths[name])
            rows[name] = {
                "path": actual["path"], "byte_equal": actual["sha256"] == expected["sha256"],
                "arrays_equal": actual.get("arrays") == expected.get("arrays"),
                "expected_sha256": expected["sha256"], "actual_sha256": actual["sha256"],
                "changed_arrays": sorted(k for k in set(expected.get("arrays", {})) | set(actual.get("arrays", {}))
                                         if expected.get("arrays", {}).get(k) != actual.get("arrays", {}).get(k)),
            }
        except (OSError, KeyError, ValueError) as error:
            rows[name] = {"byte_equal": False, "error": str(error)}
    extra = sorted(set(paths) - set(baseline["files"]))
    return {"ok": all(row["byte_equal"] for row in rows.values()) and not extra,
            "files": rows, "unexpected_labels": extra}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record")
    record.add_argument("paths", type=Path)
    record.add_argument("baseline", type=Path)
    for name in ("check", "compare"):
        check = sub.add_parser(name)
        check.add_argument("baseline", type=Path)
        if name == "compare":
            check.add_argument("paths", type=Path)
        check.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "record":
        paths = json.loads(args.paths.read_text(encoding="utf-8-sig"))
        baseline = {"version": 1, "files": {name: describe(path) for name, path in sorted(paths.items())}}
        # Exclusive creation prevents an accidental comparison from replacing the oracle.
        with args.baseline.open("x", encoding="utf-8") as stream:
            json.dump(baseline, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(json.dumps({"recorded": len(paths), "baseline": str(args.baseline)}))
        return 0
    baseline = json.loads(args.baseline.read_text(encoding="utf-8-sig"))
    paths = (json.loads(args.paths.read_text(encoding="utf-8-sig")) if args.command == "compare"
             else {name: record["path"] for name, record in baseline["files"].items()})
    report = compare(baseline, paths)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
