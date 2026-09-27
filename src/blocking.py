import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction import FeatureHasher
from normalize import feats

KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)


def make_hasher(bits):
    return FeatureHasher(n_features=2 ** bits, input_type="string", alternate_sign=False, dtype=np.float32)


def hash_rows(H, names, addrs):
    X = H.transform(feats(n, d) for n, d in zip(names, addrs)).tocsr()
    X.data[:] = 1.0
    return X


def load_pool(path, H, chunk=250000):
    ids, ctry, names, addrs, mats = [], [], [], [], []
    for ch in pd.read_csv(path, chunksize=chunk, **KW):
        n = ch.business_name.to_numpy(object); d = ch.business_address.to_numpy(object)
        ids.append(ch.entity_id.to_numpy(object)); ctry.append(ch.country.to_numpy(object))
        names.append(n); addrs.append(d); mats.append(hash_rows(H, n, d))
    return dict(ids=np.concatenate(ids), ctry=np.concatenate(ctry), name=np.concatenate(names),
                addr=np.concatenate(addrs), X=sp.vstack(mats).tocsr())


class Blocker:
    """Index a pool once; query many S1 chunks (top-K by summed idf of shared rare features)."""

    def __init__(self, Xp, cap):
        N = Xp.shape[0]
        df = np.bincount(Xp.indices, minlength=Xp.shape[1])
        w = np.zeros(Xp.shape[1], np.float32)
        ok = (df >= 1) & (df <= cap)
        w[ok] = np.log(N / df[ok]).astype(np.float32)
        P = Xp.copy(); P.data = (w[P.indices] > 0).astype(np.float32); P.eliminate_zeros()
        self.PT = P.T.tocsr(); self.w = w

    def query(self, X1, K, chunk=10000):
        W = X1.copy(); W.data = self.w[W.indices]; W.eliminate_zeros()
        n1 = W.shape[0]
        idx = np.full((n1, K), -1, np.int64); sc = np.zeros((n1, K), np.float32)
        for s in range(0, n1, chunk):
            C = (W[s:s + chunk] @ self.PT).tocsr()
            for i in range(C.shape[0]):
                lo, hi = C.indptr[i], C.indptr[i + 1]
                if lo == hi: continue
                d, ix = C.data[lo:hi], C.indices[lo:hi]
                if hi - lo > K:
                    sel = np.argpartition(-d, K)[:K]; d, ix = d[sel], ix[sel]
                o = np.argsort(-d)
                idx[s + i, :len(o)] = ix[o]; sc[s + i, :len(o)] = d[o]
        return idx, sc


def block(X1, Xp, K, cap, chunk=10000):
    return Blocker(Xp, cap).query(X1, K, chunk)
