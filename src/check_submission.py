#!/usr/bin/env python3
"""
Checks a submission file against the Phase 1 rules before uploading.

Usage:
    python3 src/check_submission.py submissions/01_history_recency.csv --evaluation data/evaluation.csv
"""
import argparse
import csv
import math
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--evaluation", default="data/evaluation.csv")
    a = ap.parse_args()

    with open(a.evaluation, newline="", encoding="utf-8") as f:
        expected = [row["rowid"] for row in csv.DictReader(f)]
    expected_set = set(expected)

    errors = []
    seen = set()
    values = []
    with open(a.submission, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header != ["rowid", "predicted_probability"]:
            errors.append(f"bad header: {header!r}")
        for n, row in enumerate(r, start=2):
            if len(row) != 2:
                errors.append(f"line {n}: expected 2 columns, got {len(row)}")
                continue
            rid, val = row
            if rid in seen:
                errors.append(f"line {n}: duplicate rowid {rid}")
            elif rid not in expected_set:
                errors.append(f"line {n}: unknown rowid {rid!r}")
            seen.add(rid)
            try:
                p = float(val)
            except ValueError:
                errors.append(f"line {n}: not a number: {val!r}")
                continue
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                errors.append(f"line {n}: value outside [0, 1]: {val!r}")
            values.append(p)
            if len(errors) > 20:
                break

    missing = len(expected_set - seen)
    if missing:
        errors.append(f"{missing:,} evaluation rowids are missing")

    if errors:
        print("INVALID submission:")
        for e in errors[:20]:
            print("  " + e)
        sys.exit(1)
    print(f"OK: {len(seen):,} rows, all {len(expected):,} evaluation rowids present once, "
          f"values in [{min(values):.6g}, {max(values):.6g}], {len(set(values)):,} distinct")


if __name__ == "__main__":
    main()
