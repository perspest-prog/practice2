"""Baseline word-form spell correction and exact source-coordinate logs."""
import json
from pathlib import Path

from text_utils import normalize, tokens


def levenshtein_matrix(left, right):
    """Unit-cost insertion, deletion, substitution; no transposition operation."""
    matrix = [list(range(len(right) + 1))]
    for i, a in enumerate(left, 1):
        row = [i]
        for j, b in enumerate(right, 1):
            row.append(min(matrix[i - 1][j] + 1, row[j - 1] + 1,
                           matrix[i - 1][j - 1] + (a != b)))
        matrix.append(row)
    return matrix


def levenshtein(left, right):
    """Rolling-row dynamic programming: O(m*n) time, O(min(m,n)) space."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        row = [i]
        for j, b in enumerate(right, 1):
            row.append(min(previous[j] + 1, row[j - 1] + 1,
                           previous[j - 1] + (a != b)))
        previous = row
    return previous[-1]


def restore_case(original, display):
    if original.isupper():
        return display.upper(), "uppercase"
    if original.istitle():
        return display.title(), "titlecase"
    if original.islower():
        return display.lower(), "lowercase"
    return display, "mixed_case_uses_dictionary_display"


def transfer_span(start, end, changes):
    """Transfer source boundaries; a boundary inside a replaced token is ambiguous."""
    if not 0 <= start <= end:
        raise ValueError("Некорректный исходный диапазон")

    def boundary(position):
        shift = 0
        last_end = 0
        for change in changes:
            a, b = change["start"], change["end"]
            if a < last_end or a >= b:
                raise ValueError("Журнал замен должен быть упорядочен и не содержать пересечений")
            last_end = b
            if a < position < b:
                raise ValueError(f"Граница {position} находится внутри заменённого токена [{a}, {b})")
            if b <= position:
                shift += len(change["new"]) - (b - a)
        return position + shift

    return boundary(start), boundary(end)


class SpellCorrector:
    def __init__(self, frequencies, details=None, threshold=1, use_frequencies=True):
        if threshold not in (1, 2):
            raise ValueError("Порог расстояния должен быть 1 или 2")
        if not isinstance(frequencies, dict) or any(
                not isinstance(k, str) or normalize(k) != k or not k
                or not isinstance(v, int) or isinstance(v, bool) or v <= 0
                for k, v in frequencies.items()):
            raise ValueError("Некорректный словарь словоформ и частот")
        self.frequencies = frequencies
        self.details = details or {}
        self.threshold = threshold
        self.use_frequencies = use_frequencies
        self.by_length = {}
        for key in frequencies:
            self.by_length.setdefault(len(key), []).append(key)

    @classmethod
    def from_files(cls, dictionary_dir, **kwargs):
        path = Path(dictionary_dir)
        frequencies = json.loads((path / "word_dictionary.json").read_text())
        details = json.loads((path / "word_details.json").read_text())
        for key in frequencies:
            if key not in details or normalize(details[key]["display"]) != key:
                raise ValueError(f"Нет корректного исходного написания для {key}")
        return cls(frequencies, details, **kwargs)

    def candidates(self, word):
        key = normalize(word)
        found = []
        for length in range(max(1, len(key) - self.threshold), len(key) + self.threshold + 1):
            for candidate in self.by_length.get(length, []):
                distance = levenshtein(key, candidate)
                if distance <= self.threshold:
                    display = self.details.get(candidate, {}).get("display", candidate)
                    replacement, case_rule = restore_case(word, display)
                    found.append(dict(key=candidate, text=replacement, distance=distance,
                                      frequency=self.frequencies[candidate], case_rule=case_rule))
        found.sort(key=lambda c: (c["distance"], -c["frequency"] if self.use_frequencies else 0, c["key"]))
        return found[:3], len(found)

    def correct(self, text, protected_spans=()):
        if not isinstance(text, str):
            raise ValueError("Запрос должен быть строкой")
        spans = sorted(tuple(span) for span in protected_spans)
        for i, (a, b) in enumerate(spans):
            if not 0 <= a < b <= len(text) or (i and a < spans[i - 1][1]):
                raise ValueError("Защищённые диапазоны должны быть корректными и неперекрывающимися")
        changes, token_results, pieces = [], [], []
        cursor, shift = 0, 0
        token_text = text.replace("\u2010", "-").replace("\u2011", "-")
        for match in tokens(token_text):
            start, end = match.start(), match.end()
            word = text[start:end]
            candidates, count = [], 0
            replacement = word
            if any(start < b and a < end for a, b in spans):
                status = "protected"
            elif normalize(word) in self.frequencies:
                status = "known"
            else:
                candidates, count = self.candidates(word)
                status = "replaced" if candidates else "no_candidates"
                if candidates:
                    replacement = candidates[0]["text"]
            new_start = start + shift
            new_end = new_start + len(replacement)
            result = dict(start=start, end=end, original=word, proposed=replacement,
                          proposed_start=new_start, proposed_end=new_end, status=status,
                          candidates=candidates, candidate_count=count, ambiguous=count > 1)
            token_results.append(result)
            if replacement != word:
                changes.append(dict(start=start, end=end, old=word, new=replacement,
                                    proposed_start=new_start, proposed_end=new_end,
                                    distance=candidates[0]["distance"], candidates=candidates,
                                    alternatives=candidates[1:], candidate_count=count, ambiguous=count > 1,
                                    case_rule=candidates[0]["case_rule"]))
            pieces.extend([text[cursor:start], replacement])
            cursor = end
            shift += len(replacement) - len(word)
        pieces.append(text[cursor:])
        return dict(original_text=text, proposed_text="".join(pieces), changes=changes, tokens=token_results,
                    protected_spans=[list(s) for s in spans],
                    source_positions="original_text", proposed_positions="proposed_text",
                    message="Пустой ввод" if not text.strip() else ("Нет русских слов для проверки" if not token_results else ""),
                    config=dict(threshold=self.threshold, use_frequencies=self.use_frequencies,
                                normalization="lowercase; ё=е; U+2010/U+2011=-", max_candidates=3))
