# Run from the repository root: .venv/bin/python experiments/lab/dataset_extract.py
# Reads data/train.csv and data/evaluation.csv RAW (no preprocessing of experiment 13: no click log,
# no history truncation or de-duplication) and writes data/cache/dataset_raw.npz for
# experiments/lab/dataset_analysis.py (DATASET.md).
#
# Per row: event index, position in the event (file order), item, label (train).
# Per event: user, timestamp, time_bucket, day_type, history items and timestamps exactly as given
# (up to 20, oldest first; -1 = padding). Opaque ids are mapped to integers in order of appearance
# (train first). Checks on the way: every event has 10 contiguous rows that agree on user,
# timestamp, time_bucket, day_type and both history fields.
import csv, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
csv.field_size_limit(10**8)
item_id, user_id = {}, {}
def iid(s):
    v = item_id.get(s)
    if v is None:
        v = item_id[s] = len(item_id)
    return v
def uid(s):
    v = user_id.get(s)
    if v is None:
        v = user_id[s] = len(user_id)
    return v

def read(path, labelled):
    t0 = time.time()
    r_ev, r_pos, r_item, r_lab = [], [], [], []
    e_user, e_ts, e_tb, e_dt, e_hl, e_hi, e_ht = [], [], [], [], [], [], []
    bad = {"rows_per_event": 0, "inconsistent": 0, "dup_candidate": 0}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        head = next(rd)
        col = {c: i for i, c in enumerate(head)}
        cur, key, pos, items_in_ev = None, None, 0, []
        for row in rd:
            ev = row[col["event_id"]]
            if ev != cur:
                if cur is not None and pos != 10:
                    bad["rows_per_event"] += 1
                if len(set(items_in_ev)) != len(items_in_ev):
                    bad["dup_candidate"] += 1
                cur, pos, items_in_ev = ev, 0, []
                key = (row[col["user_id"]], row[col["timestamp"]], row[col["time_bucket"]],
                       row[col["day_type"]], row[col["history_item_ids"]], row[col["history_timestamps"]])
                hi = [iid(x) for x in json.loads(key[4])]
                ht = json.loads(key[5])
                e_user.append(uid(key[0])); e_ts.append(int(key[1])); e_tb.append(int(key[2]))
                e_dt.append(int(key[3])); e_hl.append(len(hi))
                e_hi.append(hi + [-1] * (20 - len(hi))); e_ht.append(ht + [0] * (20 - len(ht)))
            elif (row[col["user_id"]], row[col["timestamp"]], row[col["time_bucket"]], row[col["day_type"]],
                  row[col["history_item_ids"]], row[col["history_timestamps"]]) != key:
                bad["inconsistent"] += 1
            it = iid(row[col["item_id"]])
            items_in_ev.append(it)
            r_ev.append(len(e_user) - 1); r_pos.append(pos); r_item.append(it)
            if labelled:
                r_lab.append(int(row[col["clicked"]]))
            pos += 1
        if pos != 10:
            bad["rows_per_event"] += 1
        if len(set(items_in_ev)) != len(items_in_ev):
            bad["dup_candidate"] += 1
    print(f"{path}: {len(r_ev):,} rows, {len(e_user):,} events, checks {bad} ({time.time() - t0:.0f}s)", flush=True)
    out = dict(ev=np.array(r_ev, np.int32), pos=np.array(r_pos, np.int8), item=np.array(r_item, np.int32),
               user=np.array(e_user, np.int32), ts=np.array(e_ts, np.int64), tb=np.array(e_tb, np.int8),
               dt=np.array(e_dt, np.int8), hlen=np.array(e_hl, np.int8),
               hist=np.array(e_hi, np.int32), hist_ts=np.array(e_ht, np.int64))
    if labelled:
        out["label"] = np.array(r_lab, np.int8)
    return out

tr = read(ROOT / "data/train.csv", True)
te = read(ROOT / "data/evaluation.csv", False)
np.savez(ROOT / "data/cache/dataset_raw.npz", n_items=len(item_id), n_users=len(user_id),
         **{f"tr_{k}": v for k, v in tr.items()}, **{f"te_{k}": v for k, v in te.items()})
print(f"items {len(item_id):,}, users {len(user_id):,}; wrote data/cache/dataset_raw.npz")
