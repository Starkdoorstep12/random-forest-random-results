"""Precompute LaBSE embeddings for name and address, aligned to file row order.
Row i of the output arrays corresponds to row i of the input TSV (0-indexed, in read order) —
this MUST match the row order load_pool()/pd.read_csv() produce elsewhere in the pipeline,
since predict.py/train_eval.py look embeddings up by absolute row position, not by entity_id.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True, help="path to a *_sourceN.tsv file")
ap.add_argument("--out", required=True, help="output dir; writes name_emb.f16.npy, addr_emb.f16.npy")
ap.add_argument("--model", default="sentence-transformers/LaBSE")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--chunk", type=int, default=250000)
ap.add_argument("--device", default=None, help="cuda / cpu; default auto")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()


def log(*x):
    print("[%6.0fs]" % (time.time() - T0), *x, flush=True)


from sentence_transformers import SentenceTransformer
model = SentenceTransformer(a.model, device=a.device)
log("model loaded, dim =", model.get_sentence_embedding_dimension())

KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)
n_rows = sum(1 for _ in open(a.file)) - 1
log("rows:", n_rows)

name_out = np.lib.format.open_memmap(f"{a.out}/name_emb.f16.npy", mode="w+", dtype=np.float16,
                                      shape=(n_rows, model.get_sentence_embedding_dimension()))
addr_out = np.lib.format.open_memmap(f"{a.out}/addr_emb.f16.npy", mode="w+", dtype=np.float16,
                                      shape=(n_rows, model.get_sentence_embedding_dimension()))
pos = 0
for ch in pd.read_csv(a.file, chunksize=a.chunk, **KW):
    names = ch.business_name.fillna("").tolist()
    addrs = ch.business_address.fillna("").tolist()
    ne = model.encode(names, batch_size=a.batch, show_progress_bar=False, convert_to_numpy=True,
                       normalize_embeddings=True)
    ae = model.encode(addrs, batch_size=a.batch, show_progress_bar=False, convert_to_numpy=True,
                       normalize_embeddings=True)
    name_out[pos:pos + len(ch)] = ne.astype(np.float16)
    addr_out[pos:pos + len(ch)] = ae.astype(np.float16)
    pos += len(ch)
    log(f"{pos:,}/{n_rows:,} embedded")
name_out.flush(); addr_out.flush()
assert pos == n_rows
log("DONE")
