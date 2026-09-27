"""Why does blocking miss true matches? Categorize per country (train sample)."""
import argparse, time
import numpy as np, pandas as pd
import blocking as B
from normalize import norm_tokens
from pairfeats import LEGAL, NONLAT

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../../student_resource/dataset")
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--K", type=int, default=30)
ap.add_argument("--cap", type=int, default=3000)
ap.add_argument("--bits", type=int, default=26)
ap.add_argument("--chunk", type=int, default=250000)
a = ap.parse_args()
H = B.make_hasher(a.bits)
s1 = pd.read_csv(f"{a.data}/train/train_source1.tsv", **B.KW)
gt = pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", **B.KW).set_index("source1_entity_id").matched_entity_ids
pools = {k: B.load_pool(f"{a.data}/train/train_source{k}.tsv", H, a.chunk) for k in (2, 3)}


def cat(nonlat, blank, name_ov, addr_ov, num_ov, hn1, hn2):
    if blank: return "1 pool address blank"
    if nonlat: return "2 non-Latin name"
    if name_ov == 0: return "3 Latin name, no shared name token"
    if addr_ov <= 1: return "4 name ok, <=1 shared addr token"
    if hn1 and hn2 and num_ov == 0: return "5 name ok, numbers disagree"
    return "6 other"


rows = []
for c in sorted(s1.country.unique()):
    smp = s1[s1.country == c].sample(min(a.n, (s1.country == c).sum()), random_state=0).reset_index(drop=True)
    nm1, ad1 = smp.business_name.to_numpy(object), smp.business_address.to_numpy(object)
    true = [set(x for x in gt[i].split(",") if x) for i in smp.entity_id]
    X1 = B.hash_rows(H, nm1, ad1)
    for k in (2, 3):
        gidx = np.where(pools[k]["ctry"] == c)[0]
        idx, _ = B.block(X1, pools[k]["X"][gidx], a.K, a.cap)
        pos = pd.Index(pools[k]["ids"][gidx])
        for i in range(len(smp)):
            tk = [x for x in true[i] if x.startswith(f"S{k}-")]
            if not tk: continue
            cand = set(idx[i][idx[i] >= 0].tolist())
            t1 = set(norm_tokens(nm1[i])) - LEGAL; a1 = set(norm_tokens(ad1[i]))
            n1 = {t for t in a1 if any(ch.isdigit() for ch in t)}
            for r in pos.get_indexer(tk):
                g = gidx[r]; nm, ad = pools[k]["name"][g], pools[k]["addr"][g]
                t2 = set(norm_tokens(nm)) - LEGAL; a2 = set(norm_tokens(ad))
                n2 = {t for t in a2 if any(ch.isdigit() for ch in t)}
                nonlat = len(NONLAT.findall(nm)) / max(len(nm), 1) > 0.3
                ct = cat(nonlat, ad.strip() == "", len(t1 & t2), len(a1 & a2), len(n1 & n2), bool(n1), bool(n2))
                rows.append((c, k, r in cand, ct))
    print("done", c, flush=True)

D = pd.DataFrame(rows, columns=["country", "src", "hit", "cat"])
for c, g in D.groupby("country"):
    print(f"\n=== {c}: true pairs={len(g):,}  miss rate={1 - g.hit.mean():.3f}   "
          + "  ".join(f"S{k} miss={1 - h.hit.mean():.3f}" for k, h in g.groupby("src")))
    t = g.groupby("cat").agg(n=("hit", "size"), miss_rate=("hit", lambda h: 1 - h.mean()))
    t["share_of_pairs"] = t.n / len(g)
    t["share_of_all_misses"] = t.n * t.miss_rate / max((~g.hit).sum(), 1)
    print(t[["share_of_pairs", "miss_rate", "share_of_all_misses"]].round(3).to_string())
