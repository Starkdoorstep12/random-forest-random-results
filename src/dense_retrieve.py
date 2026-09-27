"""Streaming top-M cosine-similarity retrieval against a large embedding pool, without ever
materializing the full (n_query x n_pool) similarity matrix. Embeddings are assumed
L2-normalized (as sentence-transformers' normalize_embeddings=True guarantees), so cosine
similarity is a plain dot product."""
import numpy as np
import torch


class DenseRetriever:
    def __init__(self, pool_emb, pool_block=200_000, device="cpu"):
        self.pool_emb = pool_emb
        self.pool_block = pool_block
        self.device = device
        self.n_pool = pool_emb.shape[0]

    def query(self, q_emb, M, q_block=2000):
        """q_emb: (n_q, dim). Returns idx, sim each (n_q, M); idx=-1 / sim=-inf padded if n_pool < M."""
        n_q = q_emb.shape[0]
        M_eff = min(M, self.n_pool)
        out_idx = np.full((n_q, M), -1, dtype=np.int64)
        out_sim = np.full((n_q, M), -np.inf, dtype=np.float32)
        for qs in range(0, n_q, q_block):
            qe = min(qs + q_block, n_q)
            qchunk = torch.as_tensor(np.asarray(q_emb[qs:qe], dtype=np.float32), device=self.device)
            run_idx = torch.full((qe - qs, M_eff), -1, dtype=torch.int64, device=self.device)
            run_sim = torch.full((qe - qs, M_eff), -float("inf"), dtype=torch.float32, device=self.device)
            for ps in range(0, self.n_pool, self.pool_block):
                pe = min(ps + self.pool_block, self.n_pool)
                pchunk = torch.as_tensor(np.asarray(self.pool_emb[ps:pe], dtype=np.float32), device=self.device)
                sim = qchunk @ pchunk.T
                k = min(M_eff, sim.shape[1])
                vals, idxs = torch.topk(sim, k, dim=1)
                idxs = idxs + ps
                cat_sim = torch.cat([run_sim, vals], dim=1)
                cat_idx = torch.cat([run_idx, idxs], dim=1)
                topm = torch.topk(cat_sim, M_eff, dim=1)
                run_sim = topm.values
                run_idx = torch.gather(cat_idx, 1, topm.indices)
            out_idx[qs:qe, :M_eff] = run_idx.cpu().numpy()
            out_sim[qs:qe, :M_eff] = run_sim.cpu().numpy()
        return out_idx, out_sim
