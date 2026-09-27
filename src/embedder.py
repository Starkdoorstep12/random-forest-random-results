"""Encode business name+address into a single normalized embedding per record. fp16 support:
~2.4x faster, lower memory than fp32 (measured on L40S: 7,350 vs 3,095 rows/sec at batch=512)."""
import time
import numpy as np


class Embedder:
    def __init__(self, model_name="sentence-transformers/LaBSE", device=None, batch_size=512, fp16=True):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        if fp16:
            self.model = self.model.half()
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, names, addrs, log_every=200_000, tag=""):
        texts = [f"{n} | {a}" for n, a in zip(names, addrs)]
        n = len(texts)
        if n <= log_every:
            return self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                      convert_to_numpy=True, normalize_embeddings=True).astype(np.float16)
        out = np.empty((n, self.dim), dtype=np.float16)
        t0 = time.time()
        for s in range(0, n, log_every):
            e = min(s + log_every, n)
            out[s:e] = self.model.encode(texts[s:e], batch_size=self.batch_size, show_progress_bar=False,
                                          convert_to_numpy=True, normalize_embeddings=True).astype(np.float16)
            rate = e / max(time.time() - t0, 1e-9)
            print(f"    [embedder{(' '+tag) if tag else ''}] {e:,}/{n:,} encoded  "
                  f"({rate:,.0f} rows/sec, eta {((n-e)/max(rate,1e-9)):.0f}s)", flush=True)
        return out
