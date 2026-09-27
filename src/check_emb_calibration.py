"""Does emb_sim have a DIFFERENT scale for true matches across countries/scripts?
If cross-lingual (non-Latin-name) true matches score systematically lower than same-script
true matches, a model calibrated on one country's scale will misjudge the other -- this would
directly explain why LOCO got WORSE for India once emb_sim became the dominant feature."""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="../artifacts_fp16")
a = ap.parse_args()
D = pd.read_pickle(f"{a.art}/val_dump.pkl")

D["script"] = np.where(D.nonlat_name > 0.3, "non-Latin name", "Latin name")

print(f"total candidate pairs: {len(D):,}  (true matches: {int(D.y.sum()):,})\n")
print("emb_sim distribution for TRUE MATCHES, by country x script:")
t = D[D.y].groupby(["country", "script"]).emb_sim.agg(["count", "mean", "median", "std",
                                                          lambda s: s.quantile(0.10),
                                                          lambda s: s.quantile(0.90)])
t.columns = ["n", "mean", "median", "std", "p10", "p90"]
print(t.round(3).to_string())

print("\nemb_sim distribution for FALSE candidates (non-matches), by country x script, for comparison:")
f = D[~D.y].groupby(["country", "script"]).emb_sim.agg(["count", "mean", "median"])
f.columns = ["n", "mean", "median"]
print(f.round(3).to_string())

print("\nSeparation (true-match median minus false-candidate median) -- bigger is better/easier to threshold:")
sep = t["median"] - f["median"]
print(sep.round(3).to_string())
