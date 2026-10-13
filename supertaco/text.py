"""ASCII folding for text embedded in VM scripts.

The colab CLI reads rendered scripts with the Windows locale codec (cp1252),
so non-ASCII characters mojibake on the VM and break prompt-key roundtrips
between the judge and the responses JSON. Fold typographic punctuation to
ASCII and drop anything else before embedding.
"""

_FOLD = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u00a0": " ",
    "\u2026": "...",
}


def ascii_fold(text: str) -> str:
    out = []
    for ch in text:
        if ch in _FOLD:
            out.append(_FOLD[ch])
        elif ord(ch) < 128:
            out.append(ch)
    return "".join(out)
