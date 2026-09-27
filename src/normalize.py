import os, re
from unidecode import unidecode

V2 = os.environ.get("NORM", "1") == "2"   # NORM=2 enables number/code normalization

ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "dr": "drive", "ter": "terrace", "terr": "terrace", "ln": "lane", "blvd": "boulevard",
    "ct": "court", "hwy": "highway", "pl": "place", "cir": "circle", "pkwy": "parkway",
    "sq": "square", "corp": "corporation", "inc": "incorporated", "ltd": "limited",
    "pvt": "private", "co": "company", "intl": "international", "assoc": "associates",
    "mfg": "manufacturing", "svcs": "services", "nr": "near", "opp": "opposite",
    "flr": "floor", "bldg": "building", "apt": "apartment", "ste": "suite",
}
_NULL = re.compile(r"<null>")
_NON = re.compile(r"[^a-z0-9]+")
_MIX1 = re.compile(r"^([a-z]{1,2})(\d+)$")   # l01303, b1
_MIX2 = re.compile(r"^(\d+)([a-z]{1,2})$")   # 12b, 23a


def norm_tokens(s):
    """Accent/script-stripped, lowercased, abbreviation-expanded token list."""
    s = unidecode(s).lower()
    s = _NULL.sub(" ", s).replace("&", " and ")
    if V2:
        s = s.replace("+", " and ")
    return [ABBR.get(t, t) for t in _NON.sub(" ", s).split()]


def _z(d):
    return d.lstrip("0") or "0"


def _strip(t):
    """Drop leading zeros in numeric / code tokens: 01303 -> 1303, l01303 -> l1303."""
    if t.isdigit():
        return _z(t)
    m = _MIX1.match(t)
    if m:
        return m.group(1) + _z(m.group(2))
    m = _MIX2.match(t)
    if m:
        return _z(m.group(1)) + m.group(2)
    return t


def _expand(t):
    """Also emit the letter and digit parts of short codes: l1303 -> l, 1303."""
    m = _MIX1.match(t)
    if m:
        return [t, m.group(1), m.group(2)]
    m = _MIX2.match(t)
    if m:
        return [t, m.group(1), m.group(2)]
    return [t]


def feats(name, addr):
    """Blocking features: name/address unigrams and adjacent-token bigrams."""
    nt, at = norm_tokens(name), norm_tokens(addr)
    if V2:
        at = [_strip(t) for t in at]
        au = [u for t in at for u in _expand(t)]
    else:
        au = at
    return (["n:" + t for t in nt] + ["a:" + t for t in au]
            + ["nb:%s_%s" % p for p in zip(nt, nt[1:])]
            + ["ab:%s_%s" % p for p in zip(at, at[1:])])
