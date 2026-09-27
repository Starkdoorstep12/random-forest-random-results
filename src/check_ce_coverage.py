"""Free check using already-saved val_dump.pkl: how many TRUE matches fell outside the
cross-encoder's top-N shortlist (ce_score == NaN)? Tells us if raising ce_topn is worth it,
with zero new compute."""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="../artifacts_ce")
a = ap.parse_args()
D = pd.read_pickle(f"{a.art}/val_dump.pkl")

true = D[D.y]
print(f"total true-match candidate rows: {len(true):,}")
print(f"true matches WITH a ce_score (inside shortlist): {int(true.ce_score.notna().sum()):,} "
      f"({true.ce_score.notna().mean():.2%})")
print(f"true matches MISSING a ce_score (outside shortlist -- topn too small): "
      f"{int(true.ce_score.isna().sum()):,} ({true.ce_score.isna().mean():.2%})")

print("\nby country:")
print(true.groupby("country").ce_score.apply(lambda s: f"{s.notna().mean():.2%} covered ({s.isna().sum()} missed)"))

miss = true[true.ce_score.isna()]
if len(miss):
    print(f"\nof the {len(miss)} missed true matches, their sparse-feature 'score' distribution:")
    print(D.loc[miss.index, "score"].describe())
