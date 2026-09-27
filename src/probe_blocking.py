"""Measure recall of hashed rare-token blocking on TRAIN (per country, per source)."""
import sys, time, argparse, resource
from collections import defaultdict
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction import FeatureHasher
from normalize import feats

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../../student_resource/dataset")
ap.add_argument("--n", type=int, default=30000, help="sampled S1 per country")
ap.add_argument("--K", type=int, default=30)
ap.add_argument("--cap", type=int, default=300, help="max pool df for a feature to be used")
ap.add_argument("--bits", type=int, default=26)
ap.add_argument("--chunk", type=int, default=250000)
a = ap.parse_args()
KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)
H = FeatureHasher(n_features=2 ** a.bits, input_type="string", alternate_sign=False, dtype=np.float32)
T0 = time.time()


def log(*x):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print("[%5.0fs %.1fGB]" % (time.time() - T0, rss), *x, flush=True)


def hash_rows(names, addrs):
    X = H.transform(feats(n, d) for n, d in zip(names, addrs)).tocsr()
    X.data[:] = 1.0
    return X


def load_pool(path):
    ids, mats, ctry = [], [], []
    for ch in pd.read_csv(path, chunksize=a.chunk, **KW):
        ids.append(ch.entity_id.to_numpy(object)); ctry.append(ch.country.to_numpy(object))
        mats.append(hash_rows(ch.business_name.to_numpy(object), ch.business_address.to_numpy(object)))
    return np.concatenate(ids), np.concatenate(ctry), sp.vstack(mats).tocsr()


def block(X1, Xp, K, cap, chunk=10000):
    N = Xp.shape[0]
    df = np.bincount(Xp.indices, minlength=Xp.shape[1])
    w = np.zeros(Xp.shape[1], np.float32)
    ok = (df >= 1) & (df <= cap)
    w[ok] = np.log(N / df[ok]).astype(np.float32)
    P = Xp.copy(); P.data = (w[P.indices] > 0).astype(np.float32); P.eliminate_zeros()
    PT = P.T.tocsr(); del P
    W = X1.copy(); W.data = w[W.indices]; W.eliminate_zeros()
    n1 = W.shape[0]
    idx = np.full((n1, K), -1, np.int64); sc = np.zeros((n1, K), np.float32)
    nnz_tot = 0
    for s in range(0, n1, chunk):
        C = (W[s:s + chunk] @ PT).tocsr()
        nnz_tot += C.nnz
        for i in range(C.shape[0]):
            lo, hi = C.indptr[i], C.indptr[i + 1]
            if lo == hi: continue
            d, ix = C.data[lo:hi], C.indices[lo:hi]
            if hi - lo > K:
                sel = np.argpartition(-d, K)[:K]; d, ix = d[sel], ix[sel]
            o = np.argsort(-d)
            idx[s + i, :len(o)] = ix[o]; sc[s + i, :len(o)] = d[o]
    return idx, sc, nnz_tot / n1


log("loading S1 / GT")
s1 = pd.read_csv(f"{a.data}/train/train_source1.tsv", **KW)
gt = pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", **KW).set_index("source1_entity_id").matched_entity_ids
pools = {}
for k in (2, 3):
    log(f"hashing pool S{k} ...")
    pools[k] = load_pool(f"{a.data}/train/train_source{k}.tsv")
    log(f"S{k} pool rows={len(pools[k][0]):,}")

show = defaultdict(list)   # source -> list of (s1_id, kind, pool_id, score)
for c in sorted(s1.country.unique()):
    smp = s1[s1.country == c].sample(min(a.n, (s1.country == c).sum()), random_state=0).reset_index(drop=True)
    X1 = hash_rows(smp.business_name.to_numpy(object), smp.business_address.to_numpy(object))
    true = [set(x for x in gt[i].split(",") if x) for i in smp.entity_id]
    print("\n" + "=" * 20, c, "sampled S1:", len(smp), " singletons:", sum(1 for t in true if not t))
    for k in (2, 3):
        ids, ctry, Xp = pools[k]
        m = ctry == c
        pid = ids[m]; Xc = Xp[np.where(m)[0]]
        idx, sc, avg_nnz = block(X1, Xc, a.K, a.cap)
        pos = pd.Index(pid)
        ranks, ntrue, full_ok, ent = [], 0, 0, 0
        for i, t in enumerate(true):
            tk = [x for x in t if x.startswith(f"S{k}-")]
            if not tk: continue
            ent += 1
            tr = set(pos.get_indexer(tk)); row = list(idx[i])
            rk = [row.index(r) if r in row else 10 ** 6 for r in tr]
            ranks += rk; ntrue += len(rk); full_ok += all(r < a.K for r in rk)
            if any(r >= a.K for r in rk) and len(show[f"miss{k}"]) < 6:
                for r in tr:
                    if r not in row: show[f"miss{k}"].append((smp.entity_id[i], "MISS", pid[r], 0.0))
                    break
            if i % 5000 == 0 and len(show[f"top{k}"]) < 40:
                for j in range(min(6, a.K)):
                    if idx[i, j] >= 0:
                        show[f"top{k}"].append((smp.entity_id[i], "TRUE" if pid[idx[i, j]] in t else "false",
                                                pid[idx[i, j]], float(sc[i, j])))
        ranks = np.array(ranks)
        print(f"  S{k}: true matches={ntrue:,} entities_with_matches={ent:,}  avg pool hits/S1 (df<=cap)={avg_nnz:,.0f}")
        print("      recall@ " + "  ".join(f"{q}:{(ranks < q).mean():.4f}" for q in (1, 3, 5, 10, 20, a.K)) +
              f"   all-matches-in-top{a.K}: {full_ok / max(ent, 1):.4f}")
    log(c, "done")

# fetch texts for examples
need = {k: set() for k in (1, 2, 3)}
for key, lst in show.items():
    k = int(key[-1])
    for s1id, _, pidv, _ in lst:
        need[1].add(s1id); need[k].add(pidv)
txt = {}
for k in (1, 2, 3):
    for ch in pd.read_csv(f"{a.data}/train/train_source{k}.tsv", chunksize=a.chunk, **KW):
        for r in ch[ch.entity_id.isin(need[k])].itertuples(index=False):
            txt[r.entity_id] = f"{r.business_name} | {r.business_address}"
for key in sorted(show):
    print("\n#### examples:", key)
    last = None
    for s1id, kind, pidv, score in show[key]:
        if s1id != last:
            print("  S1  ", s1id, txt.get(s1id)); last = s1id
        print(f"     {kind:5s} {score:6.1f}  {pidv} {txt.get(pidv)}")
log("DONE")
