"""Block (sparse UNION dense-embedding) + score the full test set (per-country pool embeddings, in-memory, fp16)."""
import argparse, gc, json, os, sys, time, resource, zlib
import multiprocessing as mp
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test", choices=["test", "train"])
ap.add_argument("--data", default="../../../../student_resource/dataset")
ap.add_argument("--art", default="../artifacts")
ap.add_argument("--work", default=None)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--bucket", default="")
ap.add_argument("--chunk_s1", type=int, default=20000)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--minp", type=float, default=0.2)
ap.add_argument("--chunk", type=int, default=250000)
ap.add_argument("--emb_device", default=None)
ap.add_argument("--emb_batch", type=int, default=512)
ap.add_argument("--emb_fp16", action="store_true", default=True)
ap.add_argument("--no_fp16", dest="emb_fp16", action="store_false")
ap.add_argument("--retrieve_device", default="cuda")
ap.add_argument("--ce_batch", type=int, default=512)
ap.add_argument("--ce_fp16", action="store_true", default=True)
ap.add_argument("--ce_no_fp16", dest="ce_fp16", action="store_false")
a = ap.parse_args()
cfg = json.load(open(f"{a.art}/config.json"))
if os.environ.get("NORM", "1") != cfg["norm"]:
    sys.exit("model was trained with NORM=%s: run this with NORM=%s" % (cfg["norm"], cfg["norm"]))
import lightgbm as lgb
import blocking as B
from normalize import norm_tokens
from pairfeats import prep, pair_feats, FEATS
from dense_retrieve import DenseRetriever

USE_EMB = bool(cfg.get("emb_model")) and cfg.get("M", 0) > 0
if USE_EMB:
    from embedder import Embedder
USE_CE = bool(cfg.get("ce_model"))
if USE_CE:
    from ce_rerank import load_cross_encoder, shortlist_and_score

W = a.work or f"../work_{a.split}"
os.makedirs(W, exist_ok=True)
T0 = time.time()


def log(*x):
    print("[%5.0fs %.1fGB]" % (time.time() - T0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), *x, flush=True)


def work(batch):
    n1, a1, n2, a2 = batch
    cache = {}
    out = np.empty((len(n1), len(FEATS)), np.float32)
    for i in range(len(n1)):
        k = (n1[i], a1[i]); p1 = cache.get(k)
        if p1 is None: p1 = cache[k] = prep(n1[i], a1[i])
        out[i] = pair_feats(p1, prep(n2[i], a2[i]))
    return out


if __name__ == "__main__":
    procs = mp.Pool(a.workers)
    booster = lgb.Booster(model_file=f"{a.art}/model.txt")
    H = B.make_hasher(cfg["bits"]); K = cfg["K"]; M = cfg.get("M", 0); cap = cfg["cap"]; cols = cfg["features"]
    embedder = Embedder(cfg["emb_model"], device=a.emb_device, batch_size=a.emb_batch, fp16=a.emb_fp16) if USE_EMB else None
    if embedder: log("embedder ready, dim =", embedder.dim, "fp16 =", a.emb_fp16)
    ce_model = load_cross_encoder(cfg["ce_model"], device=a.retrieve_device, fp16=a.ce_fp16) if USE_CE else None
    if ce_model: log("cross-encoder reranker ready:", cfg["ce_model"], "top-N:", cfg.get("ce_topn", 15))
    sd = f"{a.data}/{a.split}"
    s1 = pd.read_csv(f"{sd}/{a.split}_source1.tsv", **B.KW)
    log("S1 rows", len(s1), dict(s1.country.value_counts()))
    pools = {}
    for k in (2, 3):
        pools[k] = B.load_pool(f"{sd}/{a.split}_source{k}.tsv", H, a.chunk)
        log(f"pool S{k} hashed", len(pools[k]["ids"]))

    sel = np.ones(len(s1), bool)
    if a.bucket:
        Mb, r = map(int, a.bucket.split(","))
        key = np.array([zlib.crc32(" ".join(norm_tokens(x)[:3]).encode()) % Mb for x in s1.business_address])
        sel &= key == r
    log("S1 rows selected", int(sel.sum()))
    cands, processed = {}, []
    ctry = s1.country.to_numpy()
    for c in sorted(s1.country.unique()):
        rows_c = np.where(sel & (ctry == c))[0]
        if a.limit: rows_c = rows_c[:a.limit]
        gidx = {k: np.where(pools[k]["ctry"] == c)[0] for k in (2, 3)}
        blk = {k: B.Blocker(pools[k]["X"][gidx[k]], cap) for k in (2, 3)}
        pool_emb = {}
        if embedder is not None and M > 0:
            for k in (2, 3):
                log(f"embedding {c} pool S{k} ({len(gidx[k]):,} rows) ...")
                pool_emb[k] = embedder.encode(pools[k]["name"][gidx[k]], pools[k]["addr"][gidx[k]], tag=f"{c} S{k} pool")
            dr = {k: DenseRetriever(pool_emb[k], pool_block=200_000, device=a.retrieve_device) for k in (2, 3)}
            log(f"{c} pools embedded, dense retrievers ready")
        log("country", c, "S1", len(rows_c), "pools", {k: len(v) for k, v in gidx.items()})
        for st in range(0, len(rows_c), a.chunk_s1):
            chunk_abs_rows = rows_c[st:st + a.chunk_s1]
            sub = s1.iloc[chunk_abs_rows].reset_index(drop=True)
            X1 = B.hash_rows(H, sub.business_name.to_numpy(object), sub.business_address.to_numpy(object))
            q_emb = embedder.encode(sub.business_name.to_numpy(object), sub.business_address.to_numpy(object), tag=f"{c} S1 chunk") if embedder is not None else None
            all_rows = []
            for k in (2, 3):
                idx_s, sc_s = blk[k].query(X1, K)
                idx_d = sc_d = None
                if embedder is not None and M > 0:
                    idx_d, sc_d = dr[k].query(q_emb, M)
                for i in range(len(sub)):
                    seen = {}
                    for j, sc in zip(idx_s[i], sc_s[i]):
                        if j >= 0: seen[int(j)] = float(sc)
                    if idx_d is not None:
                        for j in idx_d[i]:
                            if j >= 0: seen.setdefault(int(j), np.nan)
                    for j, sc in seen.items():
                        es = float((q_emb[i].astype(np.float32) * pool_emb[k][j].astype(np.float32)).sum()) if embedder is not None else np.nan
                        all_rows.append((i, k, gidx[k][j], sc, es))
            if not all_rows:
                processed.append(sub.entity_id.to_numpy()); continue
            si, kk, pp, sc, es = zip(*all_rows)
            P = pd.DataFrame({"s": si, "src": kk, "p": pp, "score": sc, "emb_sim": es})
            P["score"] = P.groupby("s").score.transform(lambda x: x.fillna(x.min() - 1 if x.notna().any() else 0.0))
            P["rank"] = P.groupby(["s", "src"]).score.rank(ascending=False, method="first").astype(int) - 1
            P = P.sort_values(["s", "src", "rank"], kind="stable").reset_index(drop=True)
            s_arr, p_arr, k_arr = P.s.to_numpy(), P.p.to_numpy(), P.src.to_numpy()
            n2 = np.empty(len(P), object); a2 = np.empty(len(P), object); pid = np.empty(len(P), object)
            for k in (2, 3):
                m = k_arr == k
                n2[m] = pools[k]["name"][p_arr[m]]; a2[m] = pools[k]["addr"][p_arr[m]]; pid[m] = pools[k]["ids"][p_arr[m]]
            ids_sub = sub.entity_id.to_numpy()
            for i, sstr in pd.Series(pid).groupby(s_arr).agg(",".join).items():
                cands[ids_sub[i]] = sstr
            processed.append(ids_sub)
            g = P.groupby(["s", "src"]).score
            P["top"] = g.transform("max"); P["rel"] = P.score / P.top
            P["gap"] = (P.score - g.shift(-1)).fillna(P.score); P["ncand"] = g.transform("size")

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
            if ce_model is not None:
                s1_text_chunk = (sub.business_name + " | " + sub.business_address).to_numpy(object)
                pool_text = {kk2: pools[kk2]["name"].astype(object) + " | " + pools[kk2]["addr"].astype(object) for kk2 in (2, 3)}
                P["ce_score"] = shortlist_and_score(
                    ce_model, s_arr, k_arr, p_arr,
                    P["emb_sim"].to_numpy(np.float32) if "emb_sim" in P.columns else np.full(len(P), np.nan, np.float32),
                    text1_lookup=lambda s: s1_text_chunk[s], text2_lookup=lambda src, p: pool_text[src][p],
                    topn=cfg.get("ce_topn", 15), batch_size=a.ce_batch)
            n1 = sub.business_name.to_numpy(object)[s_arr]; a1 = sub.business_address.to_numpy(object)[s_arr]
            BS = 20000
            batches = [(n1[i:i + BS].tolist(), a1[i:i + BS].tolist(), n2[i:i + BS].tolist(), a2[i:i + BS].tolist())
                       for i in range(0, len(P), BS)]
            F = np.concatenate(procs.map(work, batches)) if batches else np.empty((0, len(FEATS)), np.float32)
            Xdf = pd.DataFrame(F, columns=FEATS)
            for cc in ("score", "rank", "rel", "gap", "top", "ncand"):
                Xdf[cc] = P[cc].to_numpy(np.float32)
            Xdf["src"] = (k_arr == 3).astype(np.float32)
            if embedder is not None:
                for cc in ("emb_sim", "emb_top", "emb_rel", "emb_ncand_hi", "emb_gap"):
                    Xdf[cc] = P[cc].to_numpy(np.float32)
            if ce_model is not None:
                Xdf["ce_score"] = P["ce_score"].to_numpy(np.float32)
            prob = booster.predict(Xdf[cols].to_numpy(np.float32), num_threads=a.workers)
            keep = prob >= a.minp
            pd.DataFrame({"s1_id": ids_sub[s_arr[keep]], "src": k_arr[keep], "pool_id": pid[keep],
                          "prob": prob[keep].astype(np.float32)}).to_pickle(f"{W}/scored_{c}_{st:08d}.pkl")
            log(f"  {c} {st + len(sub):,}/{len(rows_c):,} S1 done, pairs={len(P):,} kept={int(keep.sum()):,}")
        del blk
        if pool_emb: del pool_emb
        if 'dr' in dir(): del dr
        gc.collect()
    np.save(f"{W}/processed.npy", np.concatenate(processed) if processed else np.array([], object), allow_pickle=True)
    with open(f"{W}/candidate_pairs.tsv", "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in s1.entity_id:
            f.write(f"{i}\t{cands.get(i, '')}\n")
    log("wrote", f"{W}/candidate_pairs.tsv", "DONE")
