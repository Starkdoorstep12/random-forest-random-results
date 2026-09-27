"""Quick, FREE hyperparameter sensitivity check using already-saved val_dump.pkl -- no GPU,
no re-embedding. Splits the saved validation pairs into a fresh train/test split by entity
(not reusing the original split, to avoid double-dipping) and compares LightGBM configs."""
import argparse
import numpy as np, pandas as pd, lightgbm as lgb

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="../artifacts_crowd")
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
D = pd.read_pickle(f"{a.art}/val_dump.pkl")

feat_cols = [c for c in D.columns if c not in ("s", "src", "y", "prob", "keep", "pool_id", "s1_id", "country")]
print(f"rows: {len(D):,}  features: {len(feat_cols)}  positives: {int(D.y.sum()):,}")

rng = np.random.default_rng(a.seed)
ents = D.s1_id.unique()
is_test = rng.random(len(ents)) < 0.3
test_ents = set(ents[is_test])
tr = D[~D.s1_id.isin(test_ents)]
te = D[D.s1_id.isin(test_ents)]
print(f"fresh split: train={len(tr):,} pairs, test={len(te):,} pairs\n")


def quick_score(model, te):
    prob = model.predict_proba(te[feat_cols])[:, 1]
    from sklearn.metrics import average_precision_score
    return average_precision_score(te.y, prob)


configs = [
    ("current (n=400, leaves=63, lr=0.05)", dict(n_estimators=400, num_leaves=63, learning_rate=0.05)),
    ("more trees (n=800)",                   dict(n_estimators=800, num_leaves=63, learning_rate=0.05)),
    ("more leaves (leaves=127)",              dict(n_estimators=400, num_leaves=127, learning_rate=0.05)),
    ("lower lr, more trees",                  dict(n_estimators=800, num_leaves=63, learning_rate=0.025)),
    ("fewer leaves, regularized",             dict(n_estimators=400, num_leaves=31, learning_rate=0.05)),
]
for name, params in configs:
    m = lgb.LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            min_child_samples=50, n_jobs=8, verbose=-1, **params)
    m.fit(tr[feat_cols], tr.y)
    ap_score = quick_score(m, te)
    print(f"  {name:38s}  AP={ap_score:.4f}")
