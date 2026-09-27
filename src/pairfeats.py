import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from normalize import norm_tokens

LEGAL = set("incorporated limited private llc llp corporation company ltd pvt inc corp plc pc lp "
            "group sa sas sarl sasu eurl".split())
STOP = set("and the of for de la le du des les et an".split())
NONLAT = re.compile("[^\u0000-\u024F]")
ALIAS = re.compile(r"\b(?:d\.?b\.?a\.?|a/?k/?a|f/?k/?a|formerly)\b")
TLD = re.compile(r"\.(?:com|net|org|in|fr|co)\b")
DIG = re.compile(r"\d+")

FEATS = ["n_ratio", "n_sort", "n_set", "n_partial", "n_jw", "c_ratio", "c_set", "alias_best",
         "a_ratio", "a_set", "a_sort", "a_partial", "a_jac", "num_inter", "num_jac", "fnum_eq",
         "fnum_in", "a_blank", "nonlat_name", "nonlat_addr", "is_domain", "len_n1", "len_n2",
         "len_a1", "len_a2",
         "fnum_lev", "num_min_lev", "fnum_prefix", "short_sym_diff", "short_eq",
         "tok_miss1", "tok_miss2", "tok_ov"]


def prep(name, addr):
    nt, at = norm_tokens(name), norm_tokens(addr)
    core = [t for t in nt if t not in LEGAL] or nt
    at = [(t.lstrip("0") or "0") if t.isdigit() else t for t in at]
    nums = [d.lstrip("0") or "0" for t in at for d in DIG.findall(t)]   # digit runs: 1708c -> 1708
    low = name.lower()
    segs = ()
    if ALIAS.search(low):
        segs = tuple(" ".join(norm_tokens(s)) for s in ALIAS.split(low) if s.strip())
    short = frozenset(t for t in core if 2 <= len(t) <= 3 and t not in STOP)
    return (" ".join(nt), " ".join(core), " ".join(at), frozenset(at), frozenset(nums),
            nums[0] if nums else "", len(NONLAT.findall(name)) / max(len(name), 1),
            len(NONLAT.findall(addr)) / max(len(addr), 1), 1.0 if TLD.search(low) else 0.0,
            len(nt), len(at), segs, short, frozenset(core))


def pair_feats(a, b):
    n1, c1, ad1, as1, nm1, f1, _, _, _, ln1, la1, _, sh1, cs1 = a
    n2, c2, ad2, as2, nm2, f2, nl2, al2, dom2, ln2, la2, seg2, sh2, cs2 = b
    nan = np.nan
    blank = la2 == 0 or la1 == 0
    if blank:
        ar = ats = ast = ap = aj = nan
    else:
        ar = fuzz.ratio(ad1, ad2); ats = fuzz.token_set_ratio(ad1, ad2)
        ast = fuzz.token_sort_ratio(ad1, ad2); ap = fuzz.partial_ratio(ad1, ad2)
        aj = len(as1 & as2) / max(len(as1 | as2), 1)
    ninter = len(nm1 & nm2); nuni = len(nm1 | nm2)
    alias = max(fuzz.token_set_ratio(n1, s) for s in seg2) if seg2 else nan
    fl = Levenshtein.distance(f1, f2) if (f1 and f2) else nan
    fpre = float(bool(f1 and f2 and f1 != f2 and (f1.startswith(f2) or f2.startswith(f1))))
    nmin = min(Levenshtein.distance(x, y) for x in nm1 for y in nm2) if (nm1 and nm2) else nan
    return [fuzz.ratio(n1, n2), fuzz.token_sort_ratio(n1, n2), fuzz.token_set_ratio(n1, n2),
            fuzz.partial_ratio(n1, n2), 100 * JaroWinkler.normalized_similarity(n1, n2),
            fuzz.ratio(c1, c2), fuzz.token_set_ratio(c1, c2), alias,
            ar, ats, ast, ap, aj, ninter, (ninter / nuni) if nuni else nan,
            float(f1 != "" and f1 == f2), float(f1 != "" and f1 in nm2), float(blank),
            nl2, al2, dom2, ln1, ln2, la1, la2,
            fl, nmin, fpre, len(sh1 ^ sh2), float(sh1 == sh2),
            len(cs1 - cs2), len(cs2 - cs1), len(cs1 & cs2)]
