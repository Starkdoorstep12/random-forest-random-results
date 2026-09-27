"""Shared shortlist + cross-encoder reranking logic, used by both train_eval.py and predict.py.
Reranking EVERY blocking candidate is too slow (measured: ~10 hours at fp16 for the full test
set); reranking only each entity's top-N candidates by emb_sim is ~2 hours and, per our own
recall@N measurements, should still contain the true match in the vast majority of cases."""
import numpy as np
import torch
from sentence_transformers import CrossEncoder


def load_cross_encoder(model_path, device="cuda", fp16=True):
    kwargs = {"model_kwargs": {"torch_dtype": torch.float16}} if fp16 else {}
    return CrossEncoder(model_path, device=device, **kwargs)


def shortlist_and_score(ce_model, s_arr, src_arr, p_arr, emb_sim_arr, text1_lookup, text2_lookup,
                         topn=15, batch_size=512):
    """s_arr/src_arr/p_arr/emb_sim_arr: parallel arrays, one row per candidate pair (as in P).
    text1_lookup(row_s) -> str, text2_lookup(src, row_p) -> str.
    Returns: ce_score array, same length as s_arr, NaN for candidates outside the shortlist."""
    n = len(s_arr)
    ce_score = np.full(n, np.nan, dtype=np.float32)
    order = np.lexsort((-np.nan_to_num(emb_sim_arr, nan=-1e9), src_arr, s_arr))
    grp_key = s_arr[order] * 10 + src_arr[order]
    _, first_idx, counts = np.unique(grp_key, return_index=True, return_counts=True)
    rank = np.empty(n, dtype=np.int64)
    for st, cnt in zip(first_idx, counts):
        rank[order[st:st + cnt]] = np.arange(cnt)
    shortlist = np.where(rank < topn)[0]
    if len(shortlist) == 0:
        return ce_score
    pairs = [(text1_lookup(s_arr[i]), text2_lookup(src_arr[i], p_arr[i])) for i in shortlist]
    scores = ce_model.predict(pairs, batch_size=batch_size, show_progress_bar=False,
                               apply_softmax=False, convert_to_numpy=True)
    ce_score[shortlist] = scores.astype(np.float32)
    return ce_score
