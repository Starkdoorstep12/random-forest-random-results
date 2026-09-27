"""Fine-tune a multilingual cross-encoder on our own (text1, text2, label) pairs, using
blocking-derived hard negatives (see build_ce_pairs.py) -- following the Block-SCL recipe."""
import argparse
import numpy as np, pandas as pd
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="ce_pairs.pkl")
ap.add_argument("--base_model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
ap.add_argument("--out", default="../ce_model")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--val_frac", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

df = pd.read_pickle(a.pairs)
rng = np.random.default_rng(a.seed)
is_val = rng.random(len(df)) < a.val_frac
tr_df, va_df = df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
print(f"train pairs: {len(tr_df):,}  val pairs: {len(va_df):,}")

train_ds = Dataset.from_dict({"text1": tr_df.text1.tolist(), "text2": tr_df.text2.tolist(),
                               "label": tr_df.label.tolist()})
val_ds = Dataset.from_dict({"text1": va_df.text1.tolist(), "text2": va_df.text2.tolist(),
                             "label": va_df.label.tolist()})

model = CrossEncoder(a.base_model, num_labels=1)
loss = BinaryCrossEntropyLoss(model, pos_weight=None)
evaluator = CrossEncoderClassificationEvaluator(
    sentence_pairs=list(zip(va_df.text1, va_df.text2)), labels=va_df.label.tolist(), name="ce-val")

args = CrossEncoderTrainingArguments(
    output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size,
    per_device_eval_batch_size=a.batch_size, learning_rate=a.lr, warmup_steps=0.1,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, logging_steps=50,
    load_best_model_at_end=True, metric_for_best_model="eval_ce-val_average_precision", greater_is_better=True,
)
trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                               loss=loss, evaluator=evaluator)
trainer.train()
model.save_pretrained(a.out)
print("saved fine-tuned cross-encoder to", a.out)

res = evaluator(model)
print("final val evaluation:", res)
