"""Answer normalization shared by benchmark loaders."""

from __future__ import annotations

import re
import string
import unicodedata

_PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_answer(answer: object | None) -> str:
    """Return a conservative normalized answer string.

    The benchmark loaders preserve the raw answer separately; this helper only
    provides a stable normalized view for future evaluation and analysis.
    """

    if answer is None:
        return ""
    text = unicodedata.normalize("NFKC", str(answer))
    text = text.lower().translate(_PUNCT_TRANSLATION)
    return _WHITESPACE_RE.sub(" ", text).strip()
