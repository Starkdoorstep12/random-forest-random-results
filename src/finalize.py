"""Threshold + one-to-one assignment + per-source caps -> matching_results.tsv (and optional eval)."""
import argparse, glob, json, os
import numpy as np, pandas as pd
import blocking as B

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test", choices=["test", "train"])
ap.add_argument("--data", default="../../../../student_resource/dataset")
ap.add_argument("--art", default="../artifacts")
ap.add_argument("--work", default=None)
ap.add_argument("--t", type=float, default=None)
ap.add_argument("--noassign", action="store_true")
ap.add_argument("--tmax", type=float, default=0.0, help="entity gate: drop S1 whose best kept prob is below this")
ap.add_argument("--out", default="../../../output")
ap.add_argument("--eval", action="store_true")
ap.add_argument("--exclude_n", type=int, default=20000, help="train sample size used for training (excluded from eval)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
cfg = json.load(open(f"{a.art}/config.json"))
CAPS = {2: cfg["caps"]["2"], 3: cfg["caps"]["3"]}
W = a.work or f"../work_{a.split}"
S = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(f"{W}/scored_*.pkl"))], ignore_index=True)
print("scored pairs:", len(S), flush=True)


def decide(t, assign, tmax=0.0):
    d = S[S.prob >= t].sort_values("prob", ascending=False)
    if assign:
        d = d.drop_duplicates("pool_id")          # each S2/S3 record goes to its best-scoring S1 only
    d = d.assign(r=d.groupby(["s1_id", "src"]).cumcount())
    d = d[d.r < np.where(d.src == 2, CAPS[2], CAPS[3])]
    if tmax > 0:
        d = d[d.groupby("s1_id").prob.transform("max") >= tmax]
    return d


if a.eval:
    s1 = pd.read_csv(f"{a.data}/train/train_source1.tsv", **B.KW)
    gt = pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", **B.KW).set_index("source1_entity_id").matched_entity_ids
    proc = set(np.load(f"{W}/processed.npy", allow_pickle=True))
    trained = set()
    for c in sorted(s1.country.unique()):
        trained |= set(s1[s1.country == c].sample(min(a.exclude_n, (s1.country == c).sum()), random_state=a.seed).entity_id)
    E = [e for e in s1.entity_id if e in proc and e not in trained]
    ctry = dict(zip(s1.entity_id, s1.country))
    true = {e: set(x for x in gt[e].split(",") if x) for e in E}
    print(f"eval on {len(E):,} S1 entities (processed, not used for training); "
          f"singletons={sum(1 for e in E if not true[e]):,}", flush=True)

    def macro(d, ents):
        pm = d.groupby("s1_id").pool_id.agg(set).to_dict()
        tot = 0.0
        for e in ents:
            p, t_ = pm.get(e, set()), true[e]
            if not t_: tot += 1.0 if not p else 0.0; continue
            h = len(p & t_)
            if not p or h == 0: continue
            pr, rc = h / len(p), h / len(t_)
            tot += 1.25 * pr * rc / (0.25 * pr + rc)
        return tot / len(ents)

    print("\n  t     no-assign   assign")
    best = (0, None)
    for t in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        fa, fb = macro(decide(t, False), E), macro(decide(t, True), E)
        print(f"  {t:.2f}   {fa:.4f}     {fb:.4f}")
        if fb > best[0]: best = (fb, t)
    print(f"\nbest with assignment: t={best[1]} F0.5={best[0]:.4f}")
    print("\nentity gate sweep at best t (assignment on):")
    bg = (best[0], 0.0)
    for tm in [0.0, 0.8, 0.85, 0.9, 0.95, 0.98]:
        f_ = macro(decide(best[1], True, tm), E)
        print(f"  tmax={tm:.2f}  F0.5={f_:.4f}")
        if f_ > bg[0] + 1e-9: bg = (f_, tm)
    print(f"best gate: tmax={bg[1]} F0.5={bg[0]:.4f}")
    d = decide(best[1], True, bg[1])
    for c in sorted(set(ctry[e] for e in E)):
        ec = [e for e in E if ctry[e] == c]
        print(f"  {c}: F0.5={macro(d, ec):.4f}  ({len(ec):,} entities)")

    # ---- loss breakdown: where do the missing F0.5 points go?
    from collections import defaultdict
    pm = d.groupby("s1_id").pool_id.agg(set).to_dict()
    scored_sets = S.groupby("s1_id").pool_id.agg(set).to_dict()
    tab = defaultdict(lambda: [0, 0.0])
    for e in E:
        p = pm.get(e, set()); t_ = true[e]; c = ctry[e]
        if not t_:
            cat, f = ("A singleton with false merge", 0.0) if p else ("0 singleton correct", 1.0)
        else:
            h = len(p & t_)
            if not p:
                cat = ("C nothing predicted, true pair scored >=0.2" if (t_ & scored_sets.get(e, set()))
                       else "B nothing predicted, no true pair scored")
                f = 0.0
            elif h == 0:
                cat, f = "D only wrong predictions", 0.0
            else:
                pr, rc = h / len(p), h / len(t_)
                f = 1.25 * pr * rc / (0.25 * pr + rc)
                fp, fn = len(p) - h, len(t_) - h
                cat = ("H perfect" if fp == 0 and fn == 0 else "E some right, extra wrong only" if fn == 0
                       else "F some right, missing only" if fp == 0 else "G some right, wrong and missing")
        tab[(c, cat)][0] += 1; tab[(c, cat)][1] += 1 - f
    cs = sorted(set(ctry[e] for e in E)); nE = {c: sum(1 for e in E if ctry[e] == c) for c in cs}
    print("\nLOSS BREAKDOWN: share of entities / F0.5 points lost")
    print("  %-45s" % "category" + "".join("%22s" % c for c in cs))
    for cat in sorted(set(k[1] for k in tab)):
        line = "  %-45s" % cat
        for c in cs:
            n, l = tab.get((c, cat), [0, 0.0])
            line += "%22s" % f"{n / nE[c]:.3f} / {100 * l / nE[c]:.2f}"
        print(line)

else:
    t = a.t if a.t is not None else cfg["threshold"]
    d = decide(t, not a.noassign, a.tmax)
    pm = d.groupby("s1_id").pool_id.agg(",".join).to_dict()
    ids = pd.read_csv(f"{a.data}/{a.split}/{a.split}_source1.tsv", usecols=["entity_id"], **B.KW).entity_id
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for i in ids:
            f.write(f"{i}\t{pm.get(i, '')}\n")
    print(f"t={t} tmax={a.tmax} assign={not a.noassign}: {len(pm):,} of {len(ids):,} S1 entities have matches; "
          f"pairs={len(d):,} -> {a.out}/matching_results.tsv")
