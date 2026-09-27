"""Apply the one-to-one assignment rule to an existing val_dump.pkl -- no retraining, no GPU.
Uses entities.pkl's true n_true (from ground truth) so entities whose true matches were
completely MISSED by blocking/retrieval are correctly scored as failures, not false singletons."""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="../artifacts_dense")
a = ap.parse_args()
D = pd.read_pickle(f"{a.art}/val_dump.pkl")
E = pd.read_pickle(f"{a.art}/entities.pkl")
E = E[E.is_val].reset_index(drop=True)
CAPS = {2: 5, 3: 6}

def decide(t, assign, D):
    d = D[D.prob >= t].sort_values("prob", ascending=False)
    if assign:
        d = d.drop_duplicates("pool_id")
    d = d.assign(r=d.groupby(["s1_id", "src"]).cumcount())
    return d[d.r < d.src.map(CAPS)]

s1_to_true_candidates = D[D.y].groupby("s1_id").pool_id.agg(set).to_dict()
nt_map = dict(zip(E.s1_id, E.n_true))

def macro(d, ents):
    pm = d.groupby("s1_id").pool_id.agg(set).to_dict()
    tot = 0.0
    for e in ents:
        p = pm.get(e, set())
        nt = nt_map[e]
        t_ = s1_to_true_candidates.get(e, set())
        if nt == 0:
            tot += 1.0 if not p else 0.0; continue
        h = len(p & t_)
        if not p or h == 0:
            continue
        pr, rc = h / len(p), h / nt
        tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / len(ents)

ents = E.s1_id.tolist()
print(f"held-out entities: {len(ents):,}  (of which singletons: {(E.n_true == 0).sum():,})")
print("  t     no-assign   assign")
best = (0, None, False)
for t in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
    fa = macro(decide(t, False, D), ents)
    fb = macro(decide(t, True, D), ents)
    print(f"  {t:.2f}   {fa:.4f}     {fb:.4f}")
    if fa > best[0]: best = (fa, t, False)
    if fb > best[0]: best = (fb, t, True)
print(f"\nbest: t={best[1]} assign={best[2]} F0.5={best[0]:.4f}")
