"""Shared tokenization and normalization, preserving offsets in original text."""
import re

TOKEN_PATTERN = r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*"
TOKEN_RE = re.compile(TOKEN_PATTERN)


def normalize(text):
    return text.lower().replace("ё", "е").replace("\u2010", "-").replace("\u2011", "-")


def tokens(text):
    return TOKEN_RE.finditer(text)


def exact_mentions(text, form):
    """Word boundaries include all Unicode letters, numbers, and hyphens."""
    pattern = r"(?<![\w-])" + re.escape(normalize(form)) + r"(?![\w-])"
    # All replacements above preserve length for Russian text.
    return re.finditer(pattern, normalize(text))
