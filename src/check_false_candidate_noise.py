"""How many FALSE candidates score dangerously high (close to the true-match zone), by country?
This is the concrete measure of 'India's noise floor sits closer to true matches than US's' --
directly testing whether a US-only-trained threshold would let through more India false positives."""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="../artifacts_fp16")
a = ap.parse_args()
D = pd.read_pickle(f"{a.art}/val_dump.pkl")

for c in sorted(D.country.unique()):
    d = D[D.country == c]
    f = d[~d.y]  # false candidates only
    n_true = int(d.y.sum())
    print(f"\n=== {c}  (false candidates: {len(f):,}, true matches: {n_true:,})")
    for thresh in (0.70, 0.75, 0.80, 0.85, 0.90):
        share = (f.emb_sim >= thresh).mean()
        count = int((f.emb_sim >= thresh).sum())
        ratio = count / max(n_true, 1)
        print(f"  false candidates with emb_sim >= {thresh:.2f}: {count:7,} ({share:.3%} of all false)"
              f"   -- {ratio:.2f} per true match")
