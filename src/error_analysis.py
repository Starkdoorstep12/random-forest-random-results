"""Where does the model lose points? Per-category precision/recall + worst errors with text."""
import argparse
import numpy as np, pandas as pd
import blocking as B

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../../student_resource/dataset")
ap.add_argument("--art", default="../artifacts")
ap.add_argument("--nex", type=int, default=6)
a = ap.parse_args()
D = pd.read_pickle(f"{a.art}/val_dump.pkl")
E = pd.read_pickle(f"{a.art}/entities.pkl"); E = E[E.is_val]
D["cat"] = np.select([D.a_blank == 1, D.nonlat_name > 0.3, D.c_set < 60],
                     ["blank addr", "non-Latin name", "name dissimilar"], "name similar")

for c in sorted(D.country.unique()):
    d = D[D.country == c]; e = E[E.country == c]
    tp = int((d.keep & d.y).sum()); fp = int((d.keep & ~d.y).sum()); nt = int(e.n_true.sum())
    sing = e[e.n_true == 0].s1_id
    sfp = d[d.keep & d.s1_id.isin(sing)].s1_id.nunique()
    print(f"\n=== {c}: true pairs={nt:,}  in-candidates={int(d.y.sum()):,}  predicted TP={tp:,} FP={fp:,}"
          f"  final recall={tp / nt:.3f} precision={tp / max(tp + fp, 1):.3f}"
          f"  | singletons={len(sing):,} with false merge={sfp:,} ({sfp / max(len(sing), 1):.3f})")
    g = d.groupby("cat").apply(lambda x: pd.Series({
        "cand_pairs": len(x), "true": int(x.y.sum()), "recall": (x.keep & x.y).sum() / max(x.y.sum(), 1),
        "FP": int((x.keep & ~x.y).sum()), "precision": (x.keep & x.y).sum() / max(x.keep.sum(), 1)}), include_groups=False)
    print(g.round(3).to_string())

# who truly owns each false-positive record? (full ground truth)
fp_ids = set(D[D.keep & ~D.y].pool_id)
owner = {}
for ch in pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", chunksize=500000, **B.KW):
    for s1_, m_ in zip(ch.source1_entity_id, ch.matched_entity_ids):
        if m_:
            for x_ in m_.split(","):
                if x_ in fp_ids: owner[x_] = s1_
for c in sorted(D.country.unique()):
    f_ = D[(D.country == c) & D.keep & ~D.y]
    ow = f_.pool_id.map(owner)
    hi = f_.prob >= 0.9
    print(f"FP ownership {c}: FPs={len(f_):,}  record belongs to ANOTHER S1 (assignment can fix)={ow.notna().mean():.3f}"
          f"  belongs to nobody={ow.isna().mean():.3f}  | among p>=0.9: other-S1={ow[hi].notna().mean():.3f} (n={int(hi.sum()):,})")

# examples: most confident FPs and random FNs, per country
pick = []
for c in sorted(D.country.unique()):
    d = D[D.country == c]
    fps = d[d.keep & ~d.y].sort_values("prob", ascending=False).head(a.nex)
    fns = d[~d.keep & d.y].sample(min(a.nex, int((~d.keep & d.y).sum())), random_state=0)
    pick.append((c, "FALSE POSITIVE (most confident)", fps)); pick.append((c, "FALSE NEGATIVE (random)", fns))
need = set()
for _, _, x in pick:
    need |= set(x.s1_id) | set(x.pool_id) | set(x.pool_id.map(owner).dropna())
txt = {}
for k in (1, 2, 3):
    for ch in pd.read_csv(f"{a.data}/train/train_source{k}.tsv", chunksize=500000, **B.KW):
        for r in ch[ch.entity_id.isin(need)].itertuples(index=False):
            txt[r.entity_id] = f"{r.business_name} | {r.business_address}"
for c, title, x in pick:
    print(f"\n#### {c} {title}")
    for r in x.itertuples():
        ow_ = owner.get(r.pool_id)
        extra = f"\n   owner: {txt.get(ow_)}" if ow_ else ("\n   owner: NONE" if r.keep and not r.y else "")
        print(f"  S1  {txt.get(r.s1_id)}\n   -> p={r.prob:.2f} {r.pool_id} {txt.get(r.pool_id)}{extra}")
