"""Real throughput benchmark for the cross-encoder reranker, on real business text pairs.
Needed because cross-encoder cost scales with PAIR count (no reuse across comparisons),
unlike the bi-encoder -- this number decides how many candidates per entity we can afford
to rerank (all of them, a top-K shortlist, or only ambiguous ones)."""
import argparse, time
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--pairs_file", required=True)
ap.add_argument("--model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
ap.add_argument("--n", type=int, default=20000)
a = ap.parse_args()

if a.pairs_file.endswith(".pkl"):
    df = pd.read_pickle(a.pairs_file).head(a.n)
    pairs = list(zip(df.text1, df.text2))
else:
    KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)
    df = pd.read_csv(a.pairs_file, nrows=a.n, **KW)
    texts = (df.business_name + " | " + df.business_address).tolist()
    pairs = [(texts[i], texts[(i + 1) % len(texts)]) for i in range(len(texts))]

print(f"benchmarking {len(pairs):,} pairs with {a.model}\n")
for precision in ("fp32", "fp16"):
    kwargs = {"model_kwargs": {"torch_dtype": torch.float16}} if precision == "fp16" else {}
    m = CrossEncoder(a.model, device="cuda", **kwargs)
    for batch in (128, 256, 512):
        try:
            t0 = time.time()
            m.predict(pairs, batch_size=batch, show_progress_bar=False)
            dt = time.time() - t0
            print(f"  precision={precision:5s} batch={batch:4d}  {len(pairs)/dt:8,.0f} pairs/sec  ({dt:.1f}s)")
        except RuntimeError as e:
            print(f"  precision={precision:5s} batch={batch:4d}  FAILED: {str(e)[:80]}")
    del m
