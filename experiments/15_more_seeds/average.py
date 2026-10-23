#!/usr/bin/env python3
"""
Averages submissions row by row (matched by rowid): experiment 15 = mean of experiment 13's
submission (MF seeds 0-14) and this experiment's (MF seeds 15-29).

Usage:
    python3 experiments/15_more_seeds/average.py submissions/13_retune_7days.csv \
        submissions/15_seeds15_29.csv --out submissions/15_avg_13_15.csv
"""
import argparse
import csv


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        if next(r) != ["rowid", "predicted_probability"]:
            raise ValueError(f"{path}: unexpected header")
        rows = [(rid, float(p)) for rid, p in r]
    ids = [rid for rid, _ in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate rowids")
    return ids, dict(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    order, first = read(a.inputs[0])
    tables = [first] + [read(p)[1] for p in a.inputs[1:]]
    for p, t in zip(a.inputs, tables):
        if set(t) != set(order):
            raise ValueError(f"{p}: rowids differ from {a.inputs[0]}")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rowid", "predicted_probability"])
        for rid in order:
            w.writerow([rid, f"{sum(t[rid] for t in tables) / len(tables):.6g}"])
    print(f"wrote {a.out}: mean of {len(tables)} submissions, {len(order):,} rows")


if __name__ == "__main__":
    main()
