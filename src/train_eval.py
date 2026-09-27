"""Sample train S1 -> block (sparse UNION dense-embedding) -> pair features -> LightGBM -> threshold search."""
import argparse, gc, json, os, time, resource
import numpy as np, pandas as pd, lightgbm as lgb
import blocking as B
from pairfeats import prep, pair_feats, FEATS
from dense_retrieve import DenseRetriever

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../../student_resource/dataset")
ap.add_argument("--n", type=int, default=20000, help="sampled S1 per country")
ap.add_argument("--K", type=int, default=60, help="sparse-blocking top-K")
ap.add_argument("--M", type=int, default=20, help="dense-retrieval top-M (0 disables dense channel)")
ap.add_argument("--cap", type=int, default=3000)
ap.add_argument("--bits", type=int, default=26)
ap.add_argument("--chunk", type=int, default=250000)
ap.add_argument("--val", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--trees", type=int, default=400)
ap.add_argument("--outdir", default="../artifacts")
ap.add_argument("--exclude_ids", default="", help="CSV of entity_ids to exclude (avoid fine-tuning-data leakage)")
ap.add_argument("--drop", default="")
ap.add_argument("--loco", action="store_true")
ap.add_argument("--emb_model", default="sentence-transformers/LaBSE")
ap.add_argument("--emb_device", default=None)
ap.add_argument("--emb_batch", type=int, default=512)
ap.add_argument("--emb_fp16", action="store_true", default=True)
ap.add_argument("--no_fp16", dest="emb_fp16", action="store_false")
ap.add_argument("--retrieve_device", default="cuda")
ap.add_argument("--ce_model", default="", help="path to fine-tuned cross-encoder (empty = disabled)")
ap.add_argument("--ce_topn", type=int, default=15)
ap.add_argument("--ce_batch", type=int, default=512)
ap.add_argument("--ce_fp16", action="store_true", default=True)
ap.add_argument("--ce_no_fp16", dest="ce_fp16", action="store_false")
ap.add_argument("--adaptive_k", action="store_true")
ap.add_argument("--ak_kmin", type=int, default=15)
ap.add_argument("--ak_gap", type=float, default=0.15)
ap.add_argument("--reverse_block", action="store_true")
ap.add_argument("--reverse_sample", type=int, default=0)
a = ap.parse_args()
T0 = time.time()
CAPS = {2: 5, 3: 6}


def adaptive_keep(scores, kmin, kmax, gap):
    order = np.argsort(-scores)
    top1 = scores[order[0]] if len(scores) else 0
    keep = np.zeros(len(scores), dtype=bool)
    for rank, idx in enumerate(order):
        if rank >= kmax:
            break
        if rank < kmin or scores[idx] >= top1 - gap:
            keep[idx] = True
    return keep


def log(*x):
    print("[%5.0fs %.1fGB]" % (time.time() - T0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), *x, flush=True)


H = B.make_hasher(a.bits)
s1 = pd.read_csv(f"{a.data}/train/train_source1.tsv", **B.KW)
gt = pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", **B.KW).set_index("source1_entity_id").matched_entity_ids
pools = {}
for k in (2, 3):
    log(f"hashing pool S{k}")
    pools[k] = B.load_pool(f"{a.data}/train/train_source{k}.tsv", H, a.chunk)

ce_model = None
if a.ce_model:
    from ce_rerank import load_cross_encoder, shortlist_and_score
    ce_model = load_cross_encoder(a.ce_model, device=a.retrieve_device, fp16=a.ce_fp16)
    log("cross-encoder reranker ready:", a.ce_model)

embedder = None
if a.M > 0:
    from embedder import Embedder
    embedder = Embedder(a.emb_model, device=a.emb_device, batch_size=a.emb_batch, fp16=a.emb_fp16)
    log("embedder ready, dim =", embedder.dim, "fp16 =", a.emb_fp16)

s1_pool = s1
if a.exclude_ids:
    excl = set(pd.read_csv(a.exclude_ids, header=None)[0])
    s1_pool = s1[~s1.entity_id.isin(excl)]
    log(f"excluded {len(excl):,} previously-used entities; {len(s1_pool):,} remain to sample from")

samples = [s1_pool[s1_pool.country == c].sample(min(a.n, (s1_pool.country == c).sum()), random_state=a.seed)
           for c in sorted(s1_pool.country.unique())]
S = pd.concat(samples, ignore_index=True)
truth = [set(x for x in gt[i].split(",") if x) for i in S.entity_id]
n_true = np.array([len(t) for t in truth])

parts, off = [], 0
for smp in samples:
    c = smp.country.iloc[0]
    X1 = B.hash_rows(H, smp.business_name.to_numpy(object), smp.business_address.to_numpy(object))
    q_emb = None
    if embedder is not None:
        q_emb = embedder.encode(smp.business_name.to_numpy(object), smp.business_address.to_numpy(object), tag=f"{c} S1")
        log(f"  {c}: embedded {len(smp):,} S1 rows")
    for k in (2, 3):
        gidx = np.where(pools[k]["ctry"] == c)[0]
        idx_s, sc_s = B.block(X1, pools[k]["X"][gidx], a.K, a.cap)
        rev_by_s1 = None
        if a.reverse_block:
            n_pool = len(gidx)
            if a.reverse_sample and n_pool > a.reverse_sample:
                sub = np.random.default_rng(a.seed).choice(n_pool, a.reverse_sample, replace=False)
            else:
                sub = np.arange(n_pool)
            idx_r, sc_r = B.block(pools[k]["X"][gidx[sub]], X1, a.K, a.cap)
            rev_by_s1 = {}
            for jj in range(len(idx_r)):
                local_j = int(sub[jj])
                for s1_row, sc in zip(idx_r[jj], sc_r[jj]):
                    if s1_row >= 0:
                        rev_by_s1.setdefault(int(s1_row), []).append((local_j, float(sc)))
        pool_emb = None
        if embedder is not None and a.M > 0:
            log(f"  {c} S{k}: embedding pool ({len(gidx):,} rows) for dense retrieval ...")
            pool_emb = embedder.encode(pools[k]["name"][gidx], pools[k]["addr"][gidx], tag=f"{c} S{k} pool")
            log(f"  {c} S{k}: pool embedded, running dense retrieval")
            dr = DenseRetriever(pool_emb, pool_block=200_000, device=a.retrieve_device)
            idx_d, sc_d = dr.query(q_emb, a.M)
        rows = []
        for i in range(len(smp)):
            seen = {}
            if a.adaptive_k:
                valid = idx_s[i] >= 0
                if valid.any():
                    keep_mask = adaptive_keep(sc_s[i][valid], a.ak_kmin, a.K, a.ak_gap)
                    for j, sc in zip(idx_s[i][valid][keep_mask], sc_s[i][valid][keep_mask]):
                        seen[int(j)] = float(sc)
            else:
                for j, sc in zip(idx_s[i], sc_s[i]):
                    if j >= 0: seen[int(j)] = float(sc)
            if pool_emb is not None:
                for j in idx_d[i]:
                    if j >= 0: seen.setdefault(int(j), np.nan)
            if rev_by_s1 is not None and i in rev_by_s1:
                for j, sc in rev_by_s1[i]:
                    seen.setdefault(j, sc)
            for j, sc in seen.items():
                es = float((q_emb[i].astype(np.float32) * pool_emb[j].astype(np.float32)).sum()) if pool_emb is not None else np.nan
                rows.append((i, j, sc, es))
        if rows:
            si, pp, sc, es = zip(*rows)
            df = pd.DataFrame({"s": np.array(si) + off, "src": k, "p": gidx[np.array(pp)],
                                "score": sc, "emb_sim": es})
            df["score"] = df.groupby(df["s"]).score.transform(lambda x: x.fillna(x.min() - 1 if x.notna().any() else 0.0))
            df["rank"] = df.groupby("s").score.rank(ascending=False, method="first").astype(int) - 1
            parts.append(df)
        if a.M > 0 and 'dr' in dir():
            del dr
        if 'pool_emb' in dir() and pool_emb is not None:
            del pool_emb
        gc.collect()
    off += len(smp)
    log("blocked+retrieved", c)
P = pd.concat(parts, ignore_index=True).sort_values(["s", "src", "rank"], kind="stable").reset_index(drop=True)
pid = np.empty(len(P), dtype=object)
for k in (2, 3):
    m = (P.src == k).to_numpy(); pid[m] = pools[k]["ids"][P.p[m].to_numpy()]
P["y"] = np.fromiter((pid[i] in truth[s] for i, s in enumerate(P.s.to_numpy())), bool, len(P))
g = P.groupby(["s", "src"]).score
P["top"] = g.transform("max"); P["rel"] = P.score / P.top
P["gap"] = (P.score - g.shift(-1).where(g.shift(-1).notna() & (P.groupby(["s", "src"]).cumcount() < g.transform("size") - 1))).fillna(P.score)
P["ncand"] = g.transform("size")

if "emb_sim" in P.columns:
    ge = P.groupby(["s", "src"])
    P["emb_top"] = ge["emb_sim"].transform("max")
    P["emb_rel"] = np.where(P.emb_top.abs() > 1e-6, P.emb_sim / P.emb_top, 0.0)
    P["emb_ncand_hi"] = ge["emb_sim"].transform(lambda x: (x >= 0.75).sum())
    _tmp = P[["s", "src", "emb_sim"]].copy()
    _tmp["_i"] = np.arange(len(_tmp))
    _tmp = _tmp.sort_values(["s", "src", "emb_sim"], ascending=[True, True, False], kind="stable")
    _gg = _tmp.groupby(["s", "src"])["emb_sim"]
    _tmp["emb_gap"] = (_tmp["emb_sim"] - _gg.shift(-1)).fillna(_tmp["emb_sim"])
    P["emb_gap"] = _tmp.set_index("_i").sort_index()["emb_gap"].to_numpy()
log(f"pairs={len(P):,} positives={int(P.y.sum()):,} true total={int(n_true.sum()):,} "
    f"combined recall={P.y.sum() / n_true.sum():.4f}")

sv, kv, pv = P.s.to_numpy(), P.src.to_numpy(), P.p.to_numpy()
if ce_model is not None:
    log("running cross-encoder reranker on shortlist (top", a.ce_topn, "by emb_sim per source)...")
    s1_text = (S.business_name + " | " + S.business_address).to_numpy(object)
    pool_text = {k: pools[k]["name"].astype(object) + " | " + pools[k]["addr"].astype(object) for k in (2, 3)}
    ce_score = shortlist_and_score(
        ce_model, sv, kv, pv, P["emb_sim"].to_numpy(np.float32) if "emb_sim" in P.columns else np.full(len(P), np.nan, np.float32),
        text1_lookup=lambda s: s1_text[s], text2_lookup=lambda src, p: pool_text[src][p],
        topn=a.ce_topn, batch_size=a.ce_batch)
    P["ce_score"] = ce_score
    log(f"cross-encoder scored {int((~np.isnan(ce_score)).sum()):,} of {len(P):,} candidates")

s_prep = [prep(n, d) for n, d in zip(S.business_name, S.business_address)]
cache = {}
for k in (2, 3):
    for p in np.unique(P.p[P.src == k].to_numpy()):
        cache[(k, p)] = prep(pools[k]["name"][p], pools[k]["addr"][p])
log("prepped", len(cache), "pool records")
F = np.array([pair_feats(s_prep[s], cache[(k, p)]) for s, k, p in zip(sv, kv, pv)], dtype=np.float32)
X = pd.DataFrame(F, columns=FEATS)
for c in ("score", "rank", "rel", "gap", "top", "ncand"):
    X[c] = P[c].to_numpy(np.float32)
X["src"] = (P.src.to_numpy() == 3).astype(np.float32)
if embedder is not None:
    for cc in ("emb_sim", "emb_top", "emb_rel", "emb_ncand_hi", "emb_gap"):
        X[cc] = P[cc].to_numpy(np.float32)
if ce_model is not None:
    X["ce_score"] = P["ce_score"].to_numpy(np.float32)

X = X.drop(columns=[c for c in a.drop.split(",") if c])
COLS = list(X.columns)
log("features done", X.shape)


def macro_f05(sidx, prob, t, use_caps=True):
    keep = prob >= t
    if use_caps:
        r = pd.Series(np.where(keep, prob, -1.0)).groupby([P.s.to_numpy()[sidx], P.src.to_numpy()[sidx]]).rank(ascending=False, method="first")
        cap = np.where(P.src.to_numpy()[sidx] == 2, CAPS[2], CAPS[3])
        keep = keep & (r.to_numpy() <= cap)
    return keep


def eval_set(ent, rows, prob, ts):
    s_all = P.s.to_numpy()[rows]; y = P.y.to_numpy()[rows]
    out = {}
    for t in ts:
        keep = macro_f05(rows, prob, t)
        npred = np.bincount(s_all[keep], minlength=len(S))[ent]
        nhit = np.bincount(s_all[keep & y], minlength=len(S))[ent]
        nt = n_true[ent]
        with np.errstate(divide="ignore", invalid="ignore"):
            p_ = nhit / npred; r_ = nhit / nt
            f = np.where((npred > 0) & (nt > 0) & (nhit > 0), 1.25 * p_ * r_ / (0.25 * p_ + r_), 0.0)
        f = np.where((nt == 0) & (npred == 0), 1.0, f)
        out[t] = float(f.mean())
    return out


def fit(rows):
    m = lgb.LGBMClassifier(n_estimators=a.trees, learning_rate=0.05, num_leaves=63, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                           n_jobs=8, verbose=-1)
    m.fit(X.iloc[rows], P.y.to_numpy()[rows])
    return m


TS = [round(float(t), 2) for t in np.arange(0.30, 0.96, 0.05)]
rng = np.random.default_rng(a.seed)
is_val = rng.random(len(S)) < a.val
tr_rows = np.where(~is_val[P.s.to_numpy()])[0]; va_rows = np.where(is_val[P.s.to_numpy()])[0]
va_ent = np.where(is_val)[0]
model = fit(tr_rows)
prob = model.predict_proba(X.iloc[va_rows])[:, 1]
res = eval_set(va_ent, va_rows, prob, TS)
best_t = max(res, key=res.get)
print("\nVALIDATION macro-F0.5 by threshold:", {t: round(v, 4) for t, v in res.items()})
print(f"BEST t={best_t}  F0.5={res[best_t]:.4f}  (val S1 entities={len(va_ent):,})")
cty = S.country.to_numpy()
for c in sorted(set(cty)):
    e = va_ent[cty[va_ent] == c]
    rr = va_rows[cty[P.s.to_numpy()[va_rows]] == c]
    pr = prob[cty[P.s.to_numpy()[va_rows]] == c]
    print(f"  {c}: F0.5={eval_set(e, rr, pr, [best_t])[best_t]:.4f}")
imp = pd.Series(model.booster_.feature_importance("gain"), index=COLS).sort_values(ascending=False)
print("top features (gain):", {k: int(v) for k, v in imp.head(15).items()})

if a.loco:
    print("\nLEAVE-ONE-COUNTRY-OUT (train excluding this country, test ONLY on it -- proxy for France risk):")
    sp_ = P.s.to_numpy()
    for c in sorted(set(cty)):
        trr = np.where(cty[sp_] != c)[0]; ter = np.where(cty[sp_] == c)[0]
        mm = fit(trr); pr = mm.predict_proba(X.iloc[ter])[:, 1]
        e = np.where(cty == c)[0]
        rr = eval_set(e, ter, pr, TS); bt = max(rr, key=rr.get)
        print(f"  test={c}: best t={bt} F0.5={rr[bt]:.4f} | at in-country best_t={best_t}: {rr[best_t]:.4f}")

os.makedirs(a.outdir, exist_ok=True)
_keep = macro_f05(va_rows, prob, best_t)
_D = X.iloc[va_rows].copy().reset_index(drop=True)
_D["s"] = P.s.to_numpy()[va_rows]; _D["src"] = P.src.to_numpy()[va_rows]
_D["y"] = P.y.to_numpy()[va_rows]; _D["prob"] = prob; _D["keep"] = _keep
_D["pool_id"] = pid[va_rows]
_D["s1_id"] = S.entity_id.to_numpy()[_D.s]; _D["country"] = S.country.to_numpy()[_D.s]
_D.to_pickle(f"{a.outdir}/val_dump.pkl")
pd.DataFrame({"s1_id": S.entity_id.to_numpy(), "country": S.country.to_numpy(), "n_true": n_true,
              "is_val": is_val}).to_pickle(f"{a.outdir}/entities.pkl")
log("dumped validation predictions")

final = fit(np.arange(len(P)))
final.booster_.save_model(f"{a.outdir}/model.txt")
json.dump(dict(threshold=best_t, features=COLS, bits=a.bits, cap=a.cap, K=a.K, M=a.M, caps=CAPS,
               norm=os.environ.get("NORM", "1"), emb_model=a.emb_model if embedder else None,
               ce_model=a.ce_model or None, ce_topn=a.ce_topn),
          open(f"{a.outdir}/config.json", "w"))
log("saved model to", a.outdir, "DONE")
