"""Build (text1, text2, label) training pairs for cross-encoder fine-tuning:
- positives: real ground-truth matches
- hard negatives: the highest-scoring FALSE candidates from blocking, per positive --
  directly the Block-SCL recipe (arXiv 2207.02008): use blocking's own output as hard negatives,
  not random ones. This targets exactly the "crowded shared-address collision" failure mode.
"""
import argparse, time
import numpy as np, pandas as pd
import blocking as B

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../../student_resource/dataset")
ap.add_argument("--n", type=int, default=20000, help="sampled S1 per country")
ap.add_argument("--K", type=int, default=60)
ap.add_argument("--cap", type=int, default=3000)
ap.add_argument("--bits", type=int, default=26)
ap.add_argument("--chunk", type=int, default=250000)
ap.add_argument("--neg_per_pos", type=int, default=4, help="hardest negatives sampled per positive")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="ce_pairs.pkl")
ap.add_argument("--exclude_ids", default="", help="CSV of entity_ids to exclude from sampling (avoid train/eval leakage)")
ap.add_argument("--save_ids", default="", help="save the sampled entity_ids to this file")
a = ap.parse_args()
T0 = time.time()


def log(*x):
    print("[%5.0fs]" % (time.time() - T0), *x, flush=True)


H = B.make_hasher(a.bits)
id_to_row = {}
s1 = pd.read_csv(f"{a.data}/train/train_source1.tsv", **B.KW)
gt = pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", **B.KW).set_index("source1_entity_id").matched_entity_ids
pools = {}
for k in (2, 3):
    log(f"hashing pool S{k}")
    pools[k] = B.load_pool(f"{a.data}/train/train_source{k}.tsv", H, a.chunk)
    id_to_row[k] = {v: i for i, v in enumerate(pools[k]["ids"])}
    log(f"S{k} id index built ({len(id_to_row[k]):,} entries)")

s1_pool = s1
if a.exclude_ids:
    excl = set(pd.read_csv(a.exclude_ids, header=None)[0])
    s1_pool = s1[~s1.entity_id.isin(excl)]
    log(f"excluded {len(excl):,} previously-used entities; {len(s1_pool):,} remain to sample from")

samples = [s1_pool[s1_pool.country == c].sample(min(a.n, (s1_pool.country == c).sum()), random_state=a.seed)
           for c in sorted(s1_pool.country.unique())]

rows = []
n_pos = n_neg = 0
for smp in samples:
    c = smp.country.iloc[0]
    names1 = smp.business_name.to_numpy(object); addrs1 = smp.business_address.to_numpy(object)
    truth = [set(x for x in gt[i].split(",") if x) for i in smp.entity_id]
    X1 = B.hash_rows(H, names1, addrs1)
    for k in (2, 3):
        gidx = np.where(pools[k]["ctry"] == c)[0]
        idx_s, sc_s = B.block(X1, pools[k]["X"][gidx], a.K, a.cap)
        for i in range(len(smp)):
            t1 = f"{names1[i]} | {addrs1[i]}"
            tk = [x for x in truth[i] if x.startswith(f"S{k}-")]
            cand_local = idx_s[i][idx_s[i] >= 0]
            cand_sc = sc_s[i][idx_s[i] >= 0]
            cand_global = gidx[cand_local]
            cand_ids = pools[k]["ids"][cand_global]
            is_true = np.isin(cand_ids, tk)
            for tid in tk:
                pg = id_to_row[k].get(tid)
                if pg is None: continue
                t2 = f"{pools[k]['name'][pg]} | {pools[k]['addr'][pg]}"
                rows.append((t1, t2, 1.0)); n_pos += 1
            neg_mask = ~is_true
            if neg_mask.any():
                order = np.argsort(-cand_sc[neg_mask])[: a.neg_per_pos * max(len(tk), 1)]
                neg_global = cand_global[neg_mask][order]
                for ng in neg_global:
                    t2 = f"{pools[k]['name'][ng]} | {pools[k]['addr'][ng]}"
                    rows.append((t1, t2, 0.0)); n_neg += 1
    log("done", c)

df = pd.DataFrame(rows, columns=["text1", "text2", "label"])
df.to_pickle(a.out)
all_s1_ids = pd.concat([smp.entity_id for smp in samples])
if a.save_ids:
    all_s1_ids.to_csv(a.save_ids, index=False, header=False)
    log(f"saved {len(all_s1_ids):,} sampled entity_ids to {a.save_ids}")
log(f"wrote {len(df):,} pairs (pos={n_pos:,} neg={n_neg:,}, ratio 1:{n_neg/max(n_pos,1):.1f}) -> {a.out}")
