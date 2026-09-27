"""Why is each true pair missed? blocking miss vs low model score vs dropped by assignment/cap."""
import argparse, glob, json
import numpy as np, pandas as pd
import blocking as B

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../../student_resource/dataset")
ap.add_argument("--art", default="../artifacts_v2")
ap.add_argument("--work", default="../work_train")
ap.add_argument("--t", type=float, default=0.7)
ap.add_argument("--exclude_n", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
cfg = json.load(open(f"{a.art}/config.json"))
CAPS = {2: cfg["caps"]["2"], 3: cfg["caps"]["3"]}
S = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(f"{a.work}/scored_*.pkl"))], ignore_index=True)
d = S[S.prob >= a.t].sort_values("prob", ascending=False).drop_duplicates("pool_id")
d = d.assign(r=d.groupby(["s1_id", "src"]).cumcount())
d = d[d.r < np.where(d.src == 2, CAPS[2], CAPS[3])]
pm = d.groupby("s1_id").pool_id.agg(set).to_dict()
probmap = dict(zip(zip(S.s1_id, S.pool_id), S.prob))

s1 = pd.read_csv(f"{a.data}/train/train_source1.tsv", **B.KW)
gt = pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", **B.KW).set_index("source1_entity_id").matched_entity_ids
proc = set(np.load(f"{a.work}/processed.npy", allow_pickle=True))
trained = set()
for c in sorted(s1.country.unique()):
    trained |= set(s1[s1.country == c].sample(min(a.exclude_n, (s1.country == c).sum()), random_state=a.seed).entity_id)
E = [e for e in s1.entity_id if e in proc and e not in trained]
Eset = set(E); ctry = dict(zip(s1.entity_id, s1.country))
cf = pd.read_csv(f"{a.work}/candidate_pairs.tsv", **B.KW)
cf = cf[cf.source1_entity_id.isin(Eset)]
cand = {e: set(x.split(",")) if x else set() for e, x in zip(cf.source1_entity_id, cf.candidate_entity_ids)}

rows = []
for e in E:
    for m in (x for x in gt[e].split(",") if x):
        if m not in cand.get(e, ()): st = "1 blocking miss (not a candidate)"
        else:
            p = probmap.get((e, m))
            if p is None: st = "2 candidate, model prob < 0.2"
            elif p < a.t: st = "3 candidate, 0.2 <= prob < t"
            elif m in pm.get(e, ()): st = "0 found"
            else: st = "4 prob >= t, dropped (assignment/cap)"
        rows.append((ctry[e], m[:2], st))
D = pd.DataFrame(rows, columns=["country", "src", "status"])
D["col"] = D.country + " " + D.src
T = D.groupby(["status", "col"]).size().unstack(fill_value=0)
print(f"true pairs in eval slice: {len(D):,}   (t={a.t})")
print((T / T.sum()).round(3).to_string())
