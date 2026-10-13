"""ascii_fold: keep embedded script text ASCII-safe.

The colab CLI reads rendered scripts with the Windows locale codec (cp1252),
so any non-ASCII character in prompts/cfg/dataset mojibakes on the VM and
breaks prompt-key roundtrips (see test_loop cp1252 regression).
"""

from supertaco.text import ascii_fold


def test_folds_typographic_dashes_and_quotes():
    assert ascii_fold("non\u2011disclosure") == "non-disclosure"
    assert ascii_fold("em\u2014dash") == "em-dash"
    assert ascii_fold("en\u2013dash") == "en-dash"
    assert ascii_fold("\u2018quoted\u2019") == "'quoted'"
    assert ascii_fold("\u201cdouble\u201d") == '"double"'
    assert ascii_fold("no\u00a0break") == "no break"
    assert ascii_fold("ellipsis\u2026") == "ellipsis..."


def test_leaves_pure_ascii_unchanged():
    s = "plain ASCII 123 - 'quotes'"
    assert ascii_fold(s) == s


def test_drops_remaining_non_ascii():
    assert ascii_fold("caf\u00e9 na\u00efve") == "caf nave"
    assert ascii_fold("legal \u4fdd\u5bc6 clause") == "legal  clause"


def test_empty_string_passthrough():
    assert ascii_fold("") == ""
