"""Text normalisation for business names and addresses.

Everything here is language/country agnostic rule-based cleaning plus a
"consonant skeleton" encoding that lets romanised Indic script
(e.g. "praaivett limittedd" from Devanagari) meet its English spelling
("private limited") -> both become "prft lmtt".
No external data is used.
"""
import re

from unidecode import unidecode

# ----------------------------------------------------------------------------
# vocabularies
# ----------------------------------------------------------------------------
LEGAL = {
    "inc", "incorporated", "llc", "llp", "lp", "plc", "pllc", "pc", "corp",
    "corporation", "co", "company", "cos", "ltd", "limited", "pvt", "private",
    "pte", "gmbh", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "scop",
    "cie", "et", "fils", "opc", "trust", "group", "grp", "holdings",
}
# legal words we also strip for the "core" name but that are less generic
STOP = {
    "the", "and", "of", "de", "du", "des", "la", "le", "les", "l", "d",
    "mr", "mrs", "ms", "m", "s", "dr", "shri", "sri", "smt", "a", "an",
    "center", "centre", "www", "com", "net", "org", "in", "fr", "us", "http",
    "https",
}
ALIAS_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|f\s*/\s*k\s*/\s*a|fka|a\s*/\s*k\s*/\s*a|aka|formerly|trading as|t/a)\b[:\s]*|\|")
DOMAIN_RE = re.compile(r"\.(?:com|net|org|co\.in|in|co|fr|io|biz|us|info)\b")

ADDR_MAP = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "avn": "ave", "drive": "dr", "boulevard": "blvd", "bd": "blvd",
    "blv": "blvd", "lane": "ln", "court": "ct", "circle": "cir", "cr": "cir",
    "place": "pl", "parkway": "pkwy", "highway": "hwy", "terrace": "ter",
    "trail": "trl", "square": "sq", "suite": "ste", "apartment": "apt",
    "apartments": "apt", "appt": "apt", "floor": "fl", "flr": "fl",
    "building": "bldg", "north": "n", "south": "s", "east": "e", "west": "w",
    "number": "no", "num": "no", "nr": "near", "opposite": "opp",
    "sector": "sec", "mount": "mt", "fort": "ft", "saint": "st", "sainte": "st",
    "point": "pt", "expressway": "expy", "freeway": "fwy", "center": "ctr",
    "centre": "ctr", "junction": "jn", "cross": "x", "main": "main",
    "township": "twp", "county": "cnty", "district": "dist", "dt": "dist",
    "r": "rue", "imp": "impasse", "ch": "chemin", "all": "allee",
    "rte": "route", "fbg": "faubourg", "pte": "porte", "q": "quai",
    "hno": "h", "house": "h", "unit": "unit", "room": "rm", "block": "blk",
    "phase": "ph", "stage": "stg", "extension": "extn", "ext": "extn",
    "colony": "col", "nagar": "ngr", "marg": "mg", "chowk": "chk",
    "bangalore": "bengaluru", "gurgaon": "gurugram", "bombay": "mumbai",
    "madras": "chennai", "calcutta": "kolkata", "hissar": "hisar",
}
ADDR_DROP = {"null", "na", "none", "nan", "the", "of", "de", "du",
             "des", "la", "le", "les", "l", "d", "and", "et", "po", "box"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "arp", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga",
    "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od",
    "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg", "ts": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl",
    "new delhi": "dl new", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "chandigarh": "chd", "puducherry": "py", "pondicherry": "py",
    "dadra and nagar haveli": "dnh", "daman and diu": "dd", "ladakh": "la",
    "lakshadweep": "ld", "andaman and nicobar islands": "an",
    "or": "od",
}
FR_REGIONS = {
    "hauts de france": "hdf", "nouvelle aquitaine": "naq",
    "pays de la loire": "pdl", "ile de france": "idf",
    "auvergne rhone alpes": "ara", "provence alpes cote d azur": "paca",
    "grand est": "ges", "occitanie": "occ", "normandie": "nor",
    "bretagne": "bre", "bourgogne franche comte": "bfc",
    "centre val de loire": "cvl", "corse": "cor",
}
_STATE_MAPS = {"US": US_STATES, "India": IN_STATES, "France": FR_REGIONS}


def _phrase_re(mapping):
    keys = sorted(mapping, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")


_STATE_RES = {c: (_phrase_re(m), m) for c, m in _STATE_MAPS.items()}
# skeleton of each multi-letter state name -> code, used for romanised native script
_STATE_SKEL = {}  # filled after skeleton() is defined

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|e|eme|ème)\b")
_LEADZ = re.compile(r"\b0+(\d)")
_SPLIT_ALNUM = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z]{2,})")


def to_ascii(s: str) -> str:
    if s is None:
        return ""
    if s.isascii():
        return s.lower()
    return unidecode(s).lower()


# skeleton ---------------------------------------------------------------------
_SK_RULES = [
    (re.compile(r"ph"), "f"), (re.compile(r"ck"), "k"),
    (re.compile(r"c(?=[eiy])"), "s"), (re.compile(r"sh"), "s"),
    (re.compile(r"ch"), "c"), (re.compile(r"[cq]"), "k"),
    (re.compile(r"x"), "ks"), (re.compile(r"z"), "s"), (re.compile(r"w"), "v"),
    (re.compile(r"b"), "p"), (re.compile(r"d"), "t"), (re.compile(r"g"), "k"),
    (re.compile(r"[aeiouyh]"), ""),
]
_DUP = re.compile(r"(.)\1+")


def skeleton(tok: str) -> str:
    if tok.isdigit():
        return tok
    s = tok
    for rx, rep in _SK_RULES:
        s = rx.sub(rep, s)
    s = _DUP.sub(r"\1", s)
    return s or tok[:1]


for _c, _m in _STATE_MAPS.items():
    _STATE_SKEL[_c] = {}
    for _k, _v in _m.items():
        _sk = skeleton(_k.replace(" ", ""))
        if len(_sk) >= 4:
            _STATE_SKEL[_c][_sk] = _v


def _map_state_skel(toks, country):
    table = _STATE_SKEL.get(country)
    if not table:
        return toks
    out, i = [], 0
    while i < len(toks):
        for w in (3, 2, 1):
            if i + w <= len(toks):
                sk = skeleton("".join(toks[i:i + w]))
                if sk in table and not toks[i].isdigit():
                    out.append(table[sk])
                    i += w
                    break
        else:
            out.append(toks[i])
            i += 1
    return out


# names ------------------------------------------------------------------------
def _name_tokens(s: str):
    s = s.replace("&", " and ").replace("+", " and ")
    s = s.replace("'", "").replace("`", "")
    # collapse dotted initialisms (l.l.c. -> llc, e.u.r.l -> eurl)
    s = re.sub(r"\b((?:[a-z]\.){2,}[a-z]?)", lambda m: m.group(1).replace(".", ""), s)
    s = DOMAIN_RE.sub(" ", s)
    s = s.replace("www.", " ").replace("@", " ")
    return [t for t in _NON_ALNUM.split(s) if t]


# legal words as they look after romanising Indic script (matched by skeleton)
LEGAL_SKEL = {"prvt", "prft", "lmt", "lmtt", "lp", "nkrprt", "krprsn", "kmpn", "tprs"}


def _map_tok(t, tokmap):
    legal = LEGAL_BY_SKEL.get(skeleton(t))
    if legal is not None or t in LEGAL:
        return legal or t
    return tokmap.get(t, t)


def norm_name(raw: str, tokmap=None):
    """Return a dict of name representations.

    tokmap: optional learned {romanised token -> english token} dictionary
    (see translit.py), applied only to names written in a non-Latin script.
    """
    a = to_ascii(raw)
    # non-Latin *script* (Indic etc.), not just accented Latin letters
    nonlatin = raw is not None and not raw.isascii() and any(ord(ch) > 0x2FF for ch in raw)
    if nonlatin and tokmap:
        a = " ".join(_map_tok(t, tokmap) for t in _name_tokens(a))
    parts = [p for p in ALIAS_RE.split(a) if p and p.strip()]
    toks = _name_tokens(a)
    if nonlatin:
        toks = [LEGAL_BY_SKEL.get(skeleton(t), t) for t in toks]
    core = [t for t in toks if t not in LEGAL and t not in STOP]
    if not core:
        core = [t for t in toks if t not in STOP] or toks
    legal = sorted({t for t in toks if t in LEGAL})
    part_cores = []
    for p in parts:
        pt = [t for t in _name_tokens(p) if t not in LEGAL and t not in STOP]
        if pt:
            part_cores.append(" ".join(pt))
    return {
        "name_c": " ".join(toks),
        "name_k": " ".join(core),
        "name_s": " ".join(skeleton(t) for t in core),
        "name_parts": "|".join(part_cores) if len(part_cores) > 1 else "",
        "legal": " ".join(legal),
        "nonlatin": (raw is not None) and (not raw.isascii())
        and sum(ord(ch) > 0x2FF for ch in raw) > 2,
        "is_domain": bool(raw) and (("." in raw and DOMAIN_RE.search(raw.lower()) is not None)
                                    or raw.startswith("@")
                                    or (" " not in raw.strip() and len(raw) > 12)),
    }


LEGAL_BY_SKEL = {"prvt": "private", "prft": "private", "lmt": "limited",
                 "lmtt": "limited", "prpt": "private", "lp": "llp", "nkrprt": "incorporated",
                 "krprsn": "corporation", "kmpn": "company"}

# addresses ------------------------------------------------------------------------
def norm_addr(raw: str, country: str):
    a = to_ascii(raw)
    a = a.replace("'", "").replace("#", " ")
    a = _NON_ALNUM.sub(" ", a)
    a = _ORD.sub(r"\1", a)
    a = _LEADZ.sub(r"\1", a)
    a = _SPLIT_ALNUM.sub(" ", a)
    st = _STATE_RES.get(country)
    if st is not None:
        rx, mp = st
        a = rx.sub(lambda m: mp[m.group(1)], a)
    raw_toks = a.split()
    if raw is not None and not raw.isascii():
        raw_toks = _map_state_skel(raw_toks, country)
    toks = []
    for t in raw_toks:
        if t in ADDR_DROP:
            continue
        toks.append(ADDR_MAP.get(t, t))
    nums = [t for t in toks if any(ch.isdigit() for ch in t)]
    words = [t for t in toks if not any(ch.isdigit() for ch in t)]
    return {
        "addr_c": " ".join(toks),
        "addr_num": " ".join(nums),
        "addr_s": " ".join(skeleton(t) for t in words),
    }


def norm_record(args, tokmap=None):
    name, addr, country = args
    d = norm_name(name, tokmap)
    d.update(norm_addr(addr, country))
    return d
