"""Quick GPU throughput benchmark: batch size x precision, on a small real slice.
Run this the moment the GPU is free (e.g. right after train_eval.py finishes, before predict.py).
Takes ~2-5 minutes total, not hours -- uses a small slice, not the full pool."""
import argparse, time
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True, help="a real *_sourceN.tsv to sample from")
ap.add_argument("--n", type=int, default=100_000, help="rows to sample for the benchmark")
ap.add_argument("--model", default="sentence-transformers/LaBSE")
a = ap.parse_args()

KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)
df = pd.read_csv(a.file, nrows=a.n, **KW)
texts = (df.business_name + " | " + df.business_address).tolist()
print(f"benchmarking on {len(texts):,} real rows from {a.file}\n")

from sentence_transformers import SentenceTransformer
import torch

for precision in ("fp32", "fp16"):
    m = SentenceTransformer(a.model, device="cuda")
    if precision == "fp16":
        m = m.half()
    for batch in (256, 512, 1024, 2048):
        try:
            torch.cuda.synchronize()
            t0 = time.time()
            m.encode(texts, batch_size=batch, show_progress_bar=False, convert_to_numpy=True,
                      normalize_embeddings=True)
            torch.cuda.synchronize()
            dt = time.time() - t0
            rate = len(texts) / dt
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  precision={precision:5s} batch={batch:5d}  {rate:8,.0f} rows/sec   "
                  f"({dt:5.1f}s total, peak GPU mem {mem:.2f} GB)")
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError as e:
            print(f"  precision={precision:5s} batch={batch:5d}  FAILED: {str(e)[:80]}")
            torch.cuda.empty_cache()
    del m
    torch.cuda.empty_cache()

print("\nPick the fastest row that didn't fail -- that's --emb_batch (and note whether fp16 helps;")
print("if it does, I'll add a --fp16 flag to embedder.py before the next real run.)")
