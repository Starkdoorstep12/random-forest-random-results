"""Evaluate a fine-tuned cross-encoder on a genuinely held-out set of pairs (entities never
seen during training) -- this is the number that actually matters, not the leaked in-training
validation split."""
import argparse
import numpy as np, pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.metrics import average_precision_score, precision_recall_curve

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="../ce_model")
ap.add_argument("--holdout_pairs", default="ce_holdout.pkl")
a = ap.parse_args()

df = pd.read_pickle(a.holdout_pairs)
m = CrossEncoder(a.model)
scores = m.predict(list(zip(df.text1, df.text2)), batch_size=256, show_progress_bar=False)

ap_score = average_precision_score(df.label, scores)
prec, rec, thr = precision_recall_curve(df.label, scores)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_i = np.argmax(f1s)
print(f"held-out pairs: {len(df):,}  (positives: {int(df.label.sum()):,})")
print(f"average_precision: {ap_score:.4f}")
print(f"best F1: {f1s[best_i]:.4f} at threshold {thr[best_i] if best_i < len(thr) else 1.0:.4f} "
      f"(precision={prec[best_i]:.4f}, recall={rec[best_i]:.4f})")
