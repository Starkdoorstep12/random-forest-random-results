"""Categorize EVERY true pair that blocking missed in the eval slice; print examples per category."""
import argparse, random
import numpy as np, pandas as pd
import blocking as B
from normalize import norm_tokens
from pairfeats import LEGAL, NONLAT

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../../student_resource/dataset")
ap.add_argument("--work", default="../work_train")
ap.add_argument("--exclude_n", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--nex", type=int, default=3)
a = ap.parse_args()

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

total = {}; miss = []
for e in E:
    for m in (x for x in gt[e].split(",") if x):
        key = (ctry[e], m[:2]); total[key] = total.get(key, 0) + 1
        if m not in cand.get(e, ()): miss.append((e, m))
print(f"eval entities={len(E):,}  true pairs={sum(total.values()):,}  blocking misses={len(miss):,}", flush=True)

need = {x for p in miss for x in p}
txt = {}
for k in (1, 2, 3):
    for ch in pd.read_csv(f"{a.data}/train/train_source{k}.tsv", chunksize=500000, **B.KW):
        for r in ch[ch.entity_id.isin(need)].itertuples(index=False):
            txt[r.entity_id] = (r.business_name, r.business_address)


def cat(nonlat, blank, name_ov, addr_ov, num_ov, hn1, hn2):
    if blank: return "1 pool address blank"
    if nonlat: return "2 non-Latin name"
    if name_ov == 0: return "3 Latin name, no shared name token"
    if addr_ov <= 1: return "4 name ok, <=1 shared addr token"
    if hn1 and hn2 and num_ov == 0: return "5 name ok, numbers disagree"
    return "6 other (name+address look fine)"


rows, ex = [], {}
for e, m in miss:
    (n1, ad1), (nm, ad) = txt[e], txt[m]
    t1 = set(norm_tokens(n1)) - LEGAL; a1 = set(norm_tokens(ad1)); t2 = set(norm_tokens(nm)) - LEGAL; a2 = set(norm_tokens(ad))
    d1 = {t for t in a1 if any(ch.isdigit() for ch in t)}; d2 = {t for t in a2 if any(ch.isdigit() for ch in t)}
    nonlat = len(NONLAT.findall(nm)) / max(len(nm), 1) > 0.3
    ct = cat(nonlat, ad.strip() == "", len(t1 & t2), len(a1 & a2), len(d1 & d2), bool(d1), bool(d2))
    rows.append((ctry[e], m[:2], ct))
    ex.setdefault((ctry[e], ct), []).append((n1, ad1, m, nm, ad))
D = pd.DataFrame(rows, columns=["country", "src", "cat"])
if len(D):
    for c, g in D.groupby("country"):
        n_true = sum(v for (cc, _), v in total.items() if cc == c)
        print(f"\n=== {c}: misses={len(g):,} = {len(g) / n_true:.3f} of true pairs")
        t = g.groupby(["cat", "src"]).size().unstack(fill_value=0)
        t["share_of_misses"] = (t.sum(axis=1) / len(g)).round(3)
        t["as_share_of_all_true_pairs"] = (t.iloc[:, :2].sum(axis=1) / n_true).round(4)
        print(t.to_string())
    random.seed(0)
    for (c, ct), lst in sorted(ex.items()):
        print(f"\n#### {c} | {ct} | n={len(lst):,}")
        for n1, ad1, m, nm, ad in random.sample(lst, min(a.nex, len(lst))):
            print(f"  S1  {n1} | {ad1}\n   -> {m} {nm} | {ad}")
