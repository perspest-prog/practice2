"""Focused tests for edit distance, ranking and source-coordinate integrity."""
import contextlib
import io
import itertools
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path

from main import main
from spelling import SpellCorrector, levenshtein, levenshtein_matrix, transfer_span
from text_utils import normalize


def exhaustive_edit_distance(start, goal):
    """Independent shortest-path oracle for tiny strings over {a,b}."""
    queue, visited = deque([(start, 0)]), {start}
    limit = max(len(start), len(goal))
    while queue:
        text, distance = queue.popleft()
        if text == goal:
            return distance
        neighbours = {text[:i] + text[i + 1:] for i in range(len(text))}
        neighbours.update(text[:i] + c + text[i + 1:] for i in range(len(text)) for c in "ab")
        if len(text) < limit:
            neighbours.update(text[:i] + c + text[i:] for i in range(len(text) + 1) for c in "ab")
        for other in neighbours - visited:
            visited.add(other)
            queue.append((other, distance + 1))


class DistanceTests(unittest.TestCase):
    def test_operations_and_empty_strings(self):
        for left, right, expected in [("", "", 0), ("", "кот", 3), ("кот", "", 3),
                                      ("кот", "кот", 0), ("кт", "кот", 1),
                                      ("коот", "кот", 1), ("кит", "кот", 1),
                                      ("москва", "москве", 1), ("ab", "ba", 2),
                                      ("учлися", "учился", 2)]:
            with self.subTest(left=left, right=right):
                self.assertEqual(levenshtein(left, right), expected)
                self.assertEqual(levenshtein_matrix(left, right)[-1][-1], expected)

    def test_against_independent_shortest_path_oracle(self):
        strings = ["".join(x) for n in range(4) for x in itertools.product("ab", repeat=n)]
        for left in strings:
            for right in strings:
                expected = exhaustive_edit_distance(left, right)
                self.assertEqual(levenshtein(left, right), expected, (left, right))
                self.assertEqual(levenshtein_matrix(left, right)[-1][-1], expected)


class CorrectionTests(unittest.TestCase):
    def test_frequency_alphabet_and_distance_ranking(self):
        corrector = SpellCorrector({"мир": 1, "мор": 9, "море": 999}, threshold=2)
        self.assertEqual([x["key"] for x in corrector.candidates("мур")[0]], ["мор", "мир", "море"])
        without = SpellCorrector(corrector.frequencies, threshold=2, use_frequencies=False)
        self.assertEqual([x["key"] for x in without.candidates("мур")[0]], ["мир", "мор", "море"])
        tie = SpellCorrector({"мор": 1, "мир": 1})
        self.assertEqual(tie.correct("мур")["proposed_text"], "мир")

    def test_three_candidates_and_ambiguity(self):
        result = SpellCorrector({"кот": 1, "ком": 1, "кол": 1, "ков": 1}).correct("кос")
        self.assertEqual(len(result["changes"][0]["candidates"]), 3)
        self.assertEqual(result["changes"][0]["candidate_count"], 4)
        self.assertTrue(result["changes"][0]["ambiguous"])
        self.assertEqual(len(result["changes"][0]["alternatives"]), 2)

    def test_case_yo_and_punctuation(self):
        corrector = SpellCorrector({"москве": 10, "королев": 1}, {"москве": {"display": "Москве"}})
        self.assertEqual(corrector.correct("«Маскве», МАСКВЕ; маскве!")["proposed_text"], "«Москве», МОСКВЕ; москве!")
        self.assertEqual(corrector.correct("Королёв")["changes"], [])
        mixed = corrector.correct("мАсКве")
        self.assertEqual(mixed["changes"][0]["case_rule"], "mixed_case_uses_dictionary_display")
        self.assertEqual(mixed["proposed_text"], "Москве")

    def test_length_changes_and_span_transfer(self):
        corrector = SpellCorrector({"москве": 10, "пушкина": 10, "кот": 5})
        text = "Москв, Пушкина и коот!"
        result = corrector.correct(text)
        self.assertEqual(result["proposed_text"], "Москве, Пушкина и кот!")
        for change in result["changes"]:
            self.assertEqual(text[change["start"]:change["end"]], change["old"])
            self.assertEqual(result["proposed_text"][change["proposed_start"]:change["proposed_end"]], change["new"])
        a, b = transfer_span(7, 14, result["changes"])
        self.assertEqual(result["proposed_text"][a:b], "Пушкина")
        self.assertEqual(transfer_span(0, 5, result["changes"]), (0, 6))
        with self.assertRaisesRegex(ValueError, "внутри"):
            transfer_span(1, 5, result["changes"])

    def test_protected_ranges(self):
        result = SpellCorrector({"москве": 1, "кот": 1}).correct("Маскве, коот", [(0, 6)])
        self.assertEqual(result["proposed_text"], "Маскве, кот")
        self.assertEqual(result["tokens"][0]["status"], "protected")
        with self.assertRaises(ValueError):
            SpellCorrector({}).correct("текст", [(0, 4), (1, 3)])

    def test_known_real_word_error_is_not_detected(self):
        result = SpellCorrector({"пира": 1, "мира": 10}).correct("Карта пира.")
        self.assertEqual(result["tokens"][1]["status"], "known")
        self.assertEqual(result["proposed_text"], "Карта пира.")

    def test_empty_unknown_and_unicode_hyphens(self):
        corrector = SpellCorrector({"санкт-петербург": 1})
        self.assertEqual(corrector.correct("")["message"], "Пустой ввод")
        self.assertEqual(corrector.correct("123?!")["proposed_text"], "123?!")
        unknown = corrector.correct("Абраксас")
        self.assertEqual(unknown["tokens"][0]["status"], "no_candidates")
        self.assertEqual(unknown["proposed_text"], "Абраксас")
        hyphen = corrector.correct("Санкт‑Петербург")
        self.assertEqual(len(hyphen["tokens"]), 1)
        self.assertEqual(hyphen["changes"], [])
        self.assertEqual(hyphen["proposed_text"], "Санкт‑Петербург")

    def test_threshold_two_can_handle_transposition(self):
        one = SpellCorrector({"учился": 1}, threshold=1)
        two = SpellCorrector({"учился": 1}, threshold=2)
        self.assertEqual(one.correct("учлися")["changes"], [])
        self.assertEqual(two.correct("учлися")["proposed_text"], "учился")


class CLITests(unittest.TestCase):
    def test_file_input_and_original_file_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "word_dictionary.json").write_text(json.dumps({"москве": 1}))
            (root / "word_details.json").write_text(json.dumps({"москве": {"display": "Москве"}}))
            source = root / "input.json"
            source.write_text(json.dumps([{"id": "demo", "original_text": "Маскве."}]))
            before = source.read_bytes()
            output = root / "output.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["correct", "--file", str(source), "--dictionary-dir", str(root), "--output", str(output)])
            self.assertEqual(status, 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["results"][0]["proposed_text"], "Москве.")
            with contextlib.redirect_stderr(io.StringIO()):
                status = main(["correct", "--file", str(source), "--output", str(source)])
            self.assertEqual(status, 1)
            self.assertEqual(before, source.read_bytes())

    def test_missing_dictionary_is_reported(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()) as output:
            status = main(["correct", "--query", "текст", "--dictionary-dir", folder])
            self.assertEqual(status, 1)
            self.assertIn("Ошибка:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
