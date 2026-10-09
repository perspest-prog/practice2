"""Rule boundaries and pipeline behavior; no gold test-set queries are read."""
import contextlib
import io
import tempfile
import unittest

from main import main
from ner import DictionaryNER, ModelLoadError, SlovnetNER
from pipeline import process
from spelling import SpellCorrector


def fixture_ner():
    return DictionaryNER([
        dict(id="p", type="PERSON", forms=["Ломоносов", "Ломоносова", "Сергея Королёва"]),
        dict(id="o", type="ORG", forms=["Московский государственный университет",
             "Московский государственный университет имени М. В. Ломоносова", "МГУ"]),
        dict(id="l", type="LOC", forms=["Москва", "Москве", "Санкт-Петербург"]),
    ])


class DictionaryTests(unittest.TestCase):
    def test_longest_and_no_nested_entities(self):
        text = "«Московский государственный университет имени М. В. Ломоносова», МГУ."
        result = fixture_ner().recognize(text)
        self.assertEqual([e["type"] for e in result], ["ORG", "ORG"])
        self.assertEqual(result[0]["text"], "Московский государственный университет имени М. В. Ломоносова")
        for e in result:
            self.assertEqual(text[e["start"]:e["end"]], e["text"])

    def test_boundaries_case_and_yo(self):
        text = "МГУшник, МГУ-2, мгу; СЕРГЕЯ КОРОЛЕВА; Санкт‑Петербург."
        result = fixture_ner().recognize(text)
        self.assertEqual([e["text"] for e in result], ["мгу", "СЕРГЕЯ КОРОЛЕВА", "Санкт‑Петербург"])
        for e in result:
            self.assertEqual(text[e["start"]:e["end"]], e["text"])

    def test_context_rule_and_its_limit(self):
        ner = DictionaryNER([])
        result = ner.recognize("Экскурсия по городу Осташков.")
        self.assertEqual([(e["text"], e["type"], e["method"]) for e in result],
                         [("Осташков", "LOC", "city_context_rule")])
        self.assertEqual(ner.recognize("Осташков"), [])
        self.assertEqual(ner.recognize("В городе осташков."), [])
        self.assertEqual(DictionaryNER([], context_rules=False).recognize("В городе Осташков."), [])

    def test_empty_and_no_entities(self):
        self.assertEqual(fixture_ner().recognize(""), [])
        self.assertEqual(fixture_ner().recognize("Обычный запрос без названий."), [])


class PipelineTests(unittest.TestCase):
    def test_B_protects_name_but_A_changes_it(self):
        corrector = SpellCorrector({"сергей": 3, "королев": 3, "в": 10, "москве": 4},
                                  {"королев": {"display": "Королёв"}})
        ner = fixture_ner()
        text = "Сергея Королёва в Москве."
        a = process(text, corrector, ner, "A")
        b = process(text, corrector, ner, "B")
        self.assertEqual(a["proposed_text"], "Сергей Королёв в Москве.")
        self.assertEqual(b["proposed_text"], text)
        self.assertEqual(b["entities"][0]["text"], "Сергея Королёва")

    def test_correction_enables_ner(self):
        result = process("Маскве.", SpellCorrector({"москве": 1}), fixture_ner(), "A")
        self.assertEqual(result["proposed_text"], "Москве.")
        self.assertEqual(result["entities"][0]["type"], "LOC")

    def test_context_protection_can_leave_typo(self):
        result = process("В городе Маскве.", SpellCorrector({"в": 1, "городе": 1, "москве": 1}), fixture_ner(), "B")
        self.assertEqual(result["proposed_text"], "В городе Маскве.")
        self.assertEqual(result["initial_entities"][0]["text"], "Маскве")

    def test_B_no_protection_still_repeats_ner(self):
        class CountingNER:
            calls = []

            def recognize(self, text):
                self.calls.append(text)
                return fixture_ner().recognize(text)

            def metadata(self):
                return {"name": "fixture"}

        ner = CountingNER()
        result = process("Сергея Королёва.", SpellCorrector({"сергей": 1, "королев": 1}), ner, "B", False)
        self.assertEqual(len(ner.calls), 2)
        self.assertEqual(ner.calls, [result["original_text"], result["proposed_text"]])
        self.assertFalse(result["protected_spans"])

    def test_shifted_final_entities(self):
        result = process("Москв и МГУ.", SpellCorrector({"москве": 1, "и": 1, "мгу": 1}), fixture_ner(), "B")
        self.assertEqual(result["proposed_text"], "Москве и МГУ.")
        self.assertEqual(result["entities"][-1]["start"], 9)
        self.assertEqual(result["initial_entities"][-1]["start"], 8)
        self.assertEqual(result["entity_positions"], "proposed_text")


class ModelFailureTests(unittest.TestCase):
    def test_missing_packs_raise_explained_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ModelLoadError, "download_models"):
                SlovnetNER(folder)

    def test_cli_model_failure_is_readable(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()) as stream:
            status = main(["ner", "--recognizer", "model", "--model-dir", folder, "--query", "текст"])
            self.assertEqual(status, 1)
            self.assertIn("Не удалось загрузить", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
