"""Fine-tune the bi-encoder (LaBSE) itself using MultipleNegativesRankingLoss (MNRL) on
our own (text1, text2, label>0 pairs only) data -- targets retrieval RECALL directly,
unlike the cross-encoder which can only rerank candidates that already exist."""
import argparse
import numpy as np, pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="ce_pairs_100k.pkl")
ap.add_argument("--base_model", default="sentence-transformers/LaBSE")
ap.add_argument("--out", default="../biencoder_model")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--val_frac", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval_cap", type=int, default=1000, help="cap validation set size for the O(n^2) IR evaluator")
a = ap.parse_args()

df = pd.read_pickle(a.pairs)
pos = df[df.label == 1.0].reset_index(drop=True)
print(f"positive pairs available: {len(pos):,}")

rng = np.random.default_rng(a.seed)
is_val = rng.random(len(pos)) < a.val_frac
tr, va = pos[~is_val].reset_index(drop=True), pos[is_val].reset_index(drop=True)
print(f"train: {len(tr):,}  val: {len(va):,}")

train_ds = Dataset.from_dict({"anchor": tr.text1.tolist(), "positive": tr.text2.tolist()})

model = SentenceTransformer(a.base_model)
loss = MultipleNegativesRankingLoss(model)

va_eval = va.iloc[: min(len(va), a.eval_cap)]
va_queries = {str(i): t for i, t in enumerate(va_eval.text1)}
va_corpus = {str(i): t for i, t in enumerate(va_eval.text2)}
va_relevant = {str(i): {str(i)} for i in range(len(va_eval))}
evaluator = InformationRetrievalEvaluator(queries=va_queries, corpus=va_corpus, relevant_docs=va_relevant,
                                           name="biencoder-val", accuracy_at_k=[1, 5, 10, 30],
                                           show_progress_bar=False)

args = SentenceTransformerTrainingArguments(
    output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size,
    learning_rate=a.lr, warmup_steps=0.1, eval_strategy="epoch", save_strategy="epoch",
    save_total_limit=1, logging_steps=50,
)
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss, evaluator=evaluator)
trainer.train()
model.save(a.out)
print("saved fine-tuned bi-encoder to", a.out)
res = evaluator(model)
print("final val evaluation:", res)
