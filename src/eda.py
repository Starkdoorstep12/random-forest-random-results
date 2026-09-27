import sys, resource, pandas as pd
from collections import Counter

def nonascii(x):
    return x.str.contains(r"[^\x00-\x7F]", regex=True).mean()

D = sys.argv[1] if len(sys.argv) > 1 else "../../../../student_resource/dataset"
kw = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)
def chunks(p, **k):
    return pd.read_csv(p, chunksize=500_000, **kw, **k)
def mem():
    return f"[peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB]"

print("=== SIZES / COUNTRY / BLANKS / NON-ASCII (first 500k rows) ===", flush=True)
totals = {}
for split in ("train", "test"):
    for k in (1, 2, 3):
        n = nb = na = 0; cc = Counter(); first = True
        for ch in chunks(f"{D}/{split}/{split}_source{k}.tsv"):
            n += len(ch)
            cc.update(ch.country.value_counts().to_dict())
            nb += int((ch.business_name.str.strip() == "").sum())
            na += int((ch.business_address.str.strip() == "").sum())
            if first:
                print(f"  {split} S{k} non-ascii share: name="
                      f"{nonascii(ch.business_name):.4f} addr="
                      f"{nonascii(ch.business_address):.4f}")
                first = False
        totals[(split, k)] = n
        print(f"{split} S{k}: rows={n:,} name_blank={nb:,} addr_blank={na:,} countries={dict(cc)}", flush=True)
print(mem(), flush=True)

print("\n=== GROUND TRUTH ===", flush=True)
gt = pd.read_csv(f"{D}/train/train_ground_truth.tsv", **kw)
s1 = pd.read_csv(f"{D}/train/train_source1.tsv", usecols=["entity_id", "country"], **kw)
gt["country"] = gt.source1_entity_id.map(s1.set_index("entity_id").country)
m = gt.matched_entity_ids
gt["n2"] = m.str.count("S2-"); gt["n3"] = m.str.count("S3-"); gt["n"] = gt.n2 + gt.n3
print("GT rows:", len(gt), " S1 rows:", len(s1),
      " GT ids == S1 ids:", set(gt.source1_entity_id) == set(s1.entity_id))
print("singleton rate:", round((gt.n == 0).mean(), 4))
print("matches/entity:", dict(sorted(Counter(gt.n).most_common(15))))
print("S2 matches/entity:", dict(sorted(Counter(gt.n2).most_common(12))))
print("S3 matches/entity:", dict(sorted(Counter(gt.n3).most_common(12))))
print("singleton rate by country:", dict(gt.groupby("country").n.apply(lambda s: round((s == 0).mean(), 4))))
print("mean matches by country:", dict(gt.groupby("country").n.mean().round(2)))
own = Counter()
for s in m:
    if s:
        own.update(s.split(","))
print("distinct matched S2/S3 ids:", len(own), " owned by >1 S1:", sum(1 for v in own.values() if v > 1))
tot23 = totals[("train", 2)] + totals[("train", 3)]
print("S2/S3 train records never matched (approx):", tot23 - len(own), "of", tot23)
del own
print(mem(), flush=True)

print("\n=== PICK SAMPLES ===", flush=True)
cons = gt[gt.n > 0].sample(20_000, random_state=0)
groups, singles = [], []
for c in gt.country.unique():
    groups.append(gt[(gt.country == c) & (gt.n >= 3)].sample(3, random_state=1))
    singles.append(gt[(gt.country == c) & (gt.n == 0)].head(4))
groups = pd.concat(groups); singles = pd.concat(singles)
need = set(cons.source1_entity_id) | set(groups.source1_entity_id) | set(singles.source1_entity_id)
for s in pd.concat([cons.matched_entity_ids, groups.matched_entity_ids]):
    need.update(s.split(","))
need.discard("")
rec = {}
for k in (1, 2, 3):
    for ch in chunks(f"{D}/train/train_source{k}.tsv"):
        sub = ch[ch.entity_id.isin(need)]
        for r in sub.itertuples(index=False):
            rec[r.entity_id] = (r.business_name, r.business_address, r.country)
print("records fetched:", len(rec), "of", len(need), mem(), flush=True)

print("\n=== COUNTRY CONSISTENCY (20k GT sample) ===")
bad = tot = miss = 0
for r in cons.itertuples():
    for i in r.matched_entity_ids.split(","):
        if i not in rec: miss += 1; continue
        tot += 1; bad += rec[i][2] != r.country
print("matches with different country than their S1:", bad, "of", tot, " missing ids:", miss)

print("\n=== SAMPLE GROUPS ===")
for c in gt.country.unique():
    print("#" * 30, c)
    for r in groups[groups.country == c].itertuples():
        print("-" * 90)
        for i in [r.source1_entity_id] + r.matched_entity_ids.split(","):
            x = rec.get(i, ("?", "?", "?"))
            print(f"{i:14s}| {x[0]} | {x[1]}")
    print("-" * 90, "\nSINGLETONS")
    for r in singles[singles.country == c].itertuples():
        x = rec[r.source1_entity_id]
        print(f"{r.source1_entity_id:14s}| {x[0]} | {x[1]}")
print("\nDONE", mem())
