"""Merge sparse-blocking candidates with name-embedding and address-embedding dense-retrieval
candidates into one per-S1-row candidate set, computing name_sim/addr_sim for every candidate
regardless of which channel(s) found it."""
import numpy as np


def union_for_row(i, idx_s, sc_s, idx_dn, idx_da, name_emb_q, addr_emb_q, name_emb_pool, addr_emb_pool):
    seen = {}
    for j, sc in zip(idx_s[i], sc_s[i]):
        if j >= 0:
            seen[int(j)] = float(sc)
    if idx_dn is not None:
        for j in idx_dn[i]:
            if j >= 0:
                seen.setdefault(int(j), np.nan)
    if idx_da is not None:
        for j in idx_da[i]:
            if j >= 0:
                seen.setdefault(int(j), np.nan)
    out = []
    qn = name_emb_q[i].astype(np.float32)
    qa = addr_emb_q[i].astype(np.float32)
    for j, sc in seen.items():
        ns = float((qn * name_emb_pool[j].astype(np.float32)).sum())
        as_ = float((qa * addr_emb_pool[j].astype(np.float32)).sum())
        out.append((j, sc, ns, as_))
    return out
