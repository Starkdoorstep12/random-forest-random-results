"""Free diagnosis of category F (partial recall) using data ALREADY on disk from a predict.py
run: for each true match an entity is MISSING, determine exactly where it was lost --
(a) never a candidate at all (blocking/retrieval miss), (b) a candidate but scored below the
storage threshold (0.2) or never reached the cross-encoder shortlist, (c) scored but below the
decision threshold, or (d) scored above threshold but bumped out by the per-source cap."""
import argparse, glob
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../../student_resource/dataset")
ap.add_argument("--work", default="../work_train_ce25")
ap.add_argument("--art", default="../artifacts_ce25")
ap.add_argument("--t", type=float, default=0.75)
ap.add_argument("--n_sample", type=int, default=5000, help="entities to diagnose in detail")
a = ap.parse_args()
KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)

s1 = pd.read_csv(f"{a.data}/train/train_source1.tsv", **KW)
gt = pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", **KW).set_index("source1_entity_id").matched_entity_ids
processed = set(np.load(f"{a.work}/processed.npy", allow_pickle=True))

cf = pd.read_csv(f"{a.work}/candidate_pairs.tsv", **KW)
cf = cf[cf.source1_entity_id.isin(processed)]
cand = {e: set(x.split(",")) if x else set() for e, x in zip(cf.source1_entity_id, cf.candidate_entity_ids)}

S = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(f"{a.work}/scored_*.pkl"))], ignore_index=True)
prob_map = dict(zip(zip(S.s1_id, S.pool_id), S.prob))

d = S[S.prob >= a.t].sort_values("prob", ascending=False).drop_duplicates("pool_id")
d = d.assign(r=d.groupby(["s1_id", "src"]).cumcount())
kept = d[d.r < d.src.map({2: 5, 3: 6})]
predicted = kept.groupby("s1_id").pool_id.agg(set).to_dict()

rng = np.random.default_rng(0)
ents = list(processed)
rng.shuffle(ents)
counts = {"a_never_candidate": 0, "b_no_prob_or_low": 0, "c_below_threshold": 0, "d_capped_out": 0, "found": 0}
examined = 0
for e in ents:
    tk = set(x for x in gt[e].split(",") if x)
    if not tk:
        continue
    pred = predicted.get(e, set())
    missing = tk - pred
    if not missing:
        continue
    examined += 1
    if examined > a.n_sample:
        break
    c = cand.get(e, set())
    for m in missing:
        if m not in c:
            counts["a_never_candidate"] += 1
        else:
            p = prob_map.get((e, m))
            if p is None:
                counts["b_no_prob_or_low"] += 1
            elif p < a.t:
                counts["c_below_threshold"] += 1
            else:
                counts["d_capped_out"] += 1
    counts["found"] += len(tk & pred)

total_missing = sum(v for k, v in counts.items() if k != "found")
print(f"diagnosed {examined:,} partial-recall entities, {total_missing:,} missing true matches, "
      f"{counts['found']:,} found matches (for context)\n")
for k, v in counts.items():
    if k == "found": continue
    print(f"  {k:22s}: {v:6,}  ({v/max(total_missing,1):.1%})")
