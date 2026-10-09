"""Independent metric fixtures, including changed lengths and wrong types."""
import unittest

from evaluation import evaluate, score_query
from spelling import SpellCorrector


def gold(text, entities=(), typos=()):
    return dict(id="fixture", group="fixture", original_text=text,
                original_entities=list(entities), typos=list(typos))


class EvaluationTests(unittest.TestCase):
    def test_changed_length_uses_source_gold_not_correct_gold(self):
        g = gold("Москв и МГУ", [dict(start=0, end=5, text="Москв", type="LOC"),
                                  dict(start=8, end=11, text="МГУ", type="ORG")],
                 [dict(start=0, end=5, wrong="Москв", correct="Москве")])
        r = dict(id="fixture", **SpellCorrector({"москве": 1, "и": 1, "мгу": 1}).correct(g["original_text"]))
        r["entities"] = [dict(start=0, end=6, text="Москве", type="LOC"), dict(start=9, end=12, text="МГУ", type="ORG")]
        metric = evaluate([g], [r])
        self.assertEqual(metric["ner"]["tp"], 2)
        self.assertEqual(metric["spelling"]["correction_rate"], 1)
        self.assertEqual(metric["spelling"]["false_change_rate"], 0)

    def test_wrong_spelling_can_still_be_correct_entity(self):
        g = gold("Маскве", [dict(start=0, end=6, text="Маскве", type="LOC")],
                 [dict(start=0, end=6, wrong="Маскве", correct="Москве")])
        r = dict(id="fixture", **SpellCorrector({}).correct("Маскве"))
        r["entities"] = g["original_entities"]
        result = evaluate([g], [r])
        self.assertEqual(result["ner"]["f1"], 1)
        self.assertEqual(result["spelling"]["correction_rate"], 0)

    def test_wrong_type_is_fp_and_fn(self):
        g = gold("МГУ", [dict(start=0, end=3, text="МГУ", type="ORG")])
        r = dict(id="fixture", original_text="МГУ", entities=[dict(start=0, end=3, text="МГУ", type="LOC")])
        result = evaluate([g], [r], correction=False)
        self.assertEqual((result["ner"]["tp"], result["ner"]["fp"], result["ner"]["fn"]), (0, 1, 1))
        self.assertEqual(result["ner_by_type"]["ORG"]["fn"], 1)
        self.assertEqual(result["ner_by_type"]["LOC"]["fp"], 1)

    def test_micro_aggregation_not_mean_query_f1(self):
        a = gold("А Б", [dict(start=0, end=1, text="А", type="PERSON"), dict(start=2, end=3, text="Б", type="PERSON")])
        b = gold("В", [dict(start=0, end=1, text="В", type="PERSON")]); b["id"] = "b"
        results = [dict(id=a["id"], original_text=a["original_text"], entities=a["original_entities"]),
                   dict(id="b", original_text="В", entities=[])]
        metric = evaluate([a, b], results, False)
        self.assertEqual(metric["ner"]["precision"], 1)
        self.assertAlmostEqual(metric["ner"]["recall"], 2/3)
        self.assertAlmostEqual(metric["ner"]["f1"], .8)

    def test_top_three_is_not_top_one_success(self):
        g = gold("мур", typos=[dict(start=0, end=3, wrong="мур", correct="мир")])
        r = dict(id="fixture", **SpellCorrector({"мор": 9, "мир": 1}).correct("мур"), entities=[])
        metric = evaluate([g], [r])
        self.assertEqual(metric["spelling"]["correction_rate"], 0)
        self.assertEqual(metric["spelling"]["top3_rate"], 1)

    def test_case_and_yo_only_do_not_count_as_damage(self):
        g = gold("Королёв")
        r = dict(id="fixture", original_text="Королёв", proposed_text="КОРОЛЕВ", entities=[], tokens=[],
                 changes=[dict(start=0, end=7, old="Королёв", new="КОРОЛЕВ")])
        metric = evaluate([g], [r])
        self.assertEqual(metric["spelling"]["damaged_correct_tokens"], 0)

    def test_untransferable_span_is_reported_and_counted_fn(self):
        g = gold("Москв", [dict(start=1, end=5, text="оскв", type="LOC")])
        r = dict(id="fixture", **SpellCorrector({"москве": 1}).correct("Москв"), entities=[])
        metric = evaluate([g], [r])
        self.assertEqual(metric["transfer_failure_count"], 1)
        self.assertEqual(metric["ner"]["fn"], 1)
        self.assertTrue(metric["query_details"][0]["transfer_failures"][0]["reason"])

    def test_empty_denominators_and_duplicate_results(self):
        g = gold("")
        r = dict(id="fixture", **SpellCorrector({}).correct(""), entities=[])
        metric = evaluate([g], [r])
        self.assertEqual(metric["ner"]["f1"], 0)
        self.assertEqual(metric["spelling"]["false_change_rate"], 0)
        with self.assertRaises(ValueError):
            evaluate([g], [r, r])


if __name__ == "__main__":
    unittest.main()
