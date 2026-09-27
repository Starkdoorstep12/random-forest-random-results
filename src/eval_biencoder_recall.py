"""Compare BASE LaBSE vs the FINE-TUNED bi-encoder on genuinely held-out entities.
Encodes ALL unique texts ONCE in large batches (not per-anchor in a loop -- that
pattern caused a real, caught-before-running performance risk on a 471M-param
model), then does per-anchor ranking lookups against the precomputed matrix."""
import argparse
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--holdout", default="ce_holdout_100k.pkl")
ap.add_argument("--base_model", default="sentence-transformers/LaBSE")
ap.add_argument("--finetuned_model", default="../biencoder_model")
ap.add_argument("--ks", default="1,5,10,30")
ap.add_argument("--batch_size", type=int, default=256)
a = ap.parse_args()
ks = [int(x) for x in a.ks.split(",")]

df = pd.read_pickle(a.holdout)
df = df[df.groupby("text1").label.transform("sum") > 0].reset_index(drop=True)
anchors = df.text1.unique()
anchor_idx = {t: i for i, t in enumerate(anchors)}

def eval_model(model_path, tag):
    m = SentenceTransformer(model_path)
    emb_anchors = m.encode(list(anchors), batch_size=a.batch_size, show_progress_bar=False)
    emb_anchors = emb_anchors / (np.linalg.norm(emb_anchors, axis=1, keepdims=True) + 1e-9)
    unique_cands = df.text2.unique()
    cand_idx = {t: i for i, t in enumerate(unique_cands)}
    emb_cands = m.encode(list(unique_cands), batch_size=a.batch_size, show_progress_bar=False)
    emb_cands = emb_cands / (np.linalg.norm(emb_cands, axis=1, keepdims=True) + 1e-9)

    ranks = []
    for anchor_text, g in df.groupby("text1"):
        ai = anchor_idx[anchor_text]
        cidx = [cand_idx[t] for t in g.text2]
        sims = emb_cands[cidx] @ emb_anchors[ai]
        order = np.argsort(-sims)
        labels = g.label.to_numpy()[order]
        true_pos = np.where(labels == 1)[0]
        if len(true_pos):
            ranks.append(true_pos[0])
    ranks = np.array(ranks)
    print(f"\n{tag}: {len(ranks):,} anchors evaluated")
    for k in ks:
        print(f"  recall@{k}: {(ranks < k).mean():.4f}")
    return ranks

r_base = eval_model(a.base_model, "BASE (LaBSE, untouched)")
r_ft = eval_model(a.finetuned_model, "FINE-TUNED (biencoder_model)")

print("\n=== SUMMARY: did fine-tuning improve retrieval recall? ===")
for k in ks:
    b = (r_base < k).mean(); f = (r_ft < k).mean()
    print(f"  recall@{k}: base={b:.4f}  finetuned={f:.4f}  delta={f-b:+.4f}")
