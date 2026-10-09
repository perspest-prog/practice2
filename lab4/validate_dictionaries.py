"""Check counts, provenance, reproducibility, and exclusion of test inputs."""
import json
import re
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from build_dictionaries import ROOT, build, digest, read_json
from text_utils import exact_mentions, normalize


def main():
    directory = ROOT / "data/dictionaries"
    names = ["word_dictionary.json", "word_details.json", "entity_dictionary.json",
             "entity_provenance.json", "build_metadata.json"]
    stored = [read_json(directory / name) for name in names]
    normal_open = Path.open
    accessed = set()

    def restricted_open(path, *args, **kwargs):
        if "queries_test" in str(path) or path.name == "prepare_queries.py":
            raise AssertionError(f"Forbidden test-containing input: {path}")
        accessed.add(str(path.relative_to(ROOT)))
        return normal_open(path, *args, **kwargs)

    with patch.object(Path, "open", restricted_open):
        rebuilt = build()
    assert stored == list(rebuilt), "Output differs from deterministic rebuild"
    words, details, entities, provenance, metadata = stored
    manifest = read_json(ROOT / "data/corpus/manifest.json")
    texts = {d["id"]: (ROOT / d["text_file"]).read_text() for d in manifest}
    # Independent recount from original corpus, without the builder's tokenizer.
    expected = Counter()
    for text in texts.values():
        for token in re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*", text):
            expected[token.lower().replace("ё", "е")] += 1
    assert words == dict(expected)
    assert sum(words.values()) == metadata["corpus_tokens"] == 3789
    assert "университет" in words and "университета" in words
    assert "в" in words and "и" in words  # Stopwords are retained.
    for key, value in details.items():
        assert key == normalize(key) == normalize(value["display"])
        assert value["frequency"] == words[key]
        assert sum(value["observed_spellings"].values()) == words[key]
        assert sum(value["document_frequencies"].values()) == words[key]
        assert all(normalize(form) == key for form in value["observed_spellings"])
    dev = read_json(ROOT / "data/queries/queries_dev.json")
    queries = {q["id"]: q for q in dev}
    forms_by_type = set()
    ownership = {}
    for e in entities:
        assert e["type"] in {"PERSON", "ORG", "LOC"}
        assert normalize(e["name"]) in {normalize(f) for f in e["forms"]}
        assert len(e["forms"]) == len({normalize(f) for f in e["forms"]})
        origin = provenance[e["id"]]
        assert [f["form"] for f in origin["forms"]] == e["forms"]
        for evidence in origin["base_evidence"]:
            text = texts[evidence["document_id"]]
            assert text[evidence["start"]:evidence["end"]] == evidence["text"]
        for f in origin["forms"]:
            assert f["key"] == normalize(f["form"])
            forms_by_type.add((f["key"], e["type"]))
            assert f["key"] not in ownership or ownership[f["key"]] == e["id"]
            ownership[f["key"]] = e["id"]
            assert f["method"] == ("attested" if f["attestations"] else "manual_form")
            for a in f["attestations"]:
                assert a["kind"] in {"corpus", "dev"}
                text = texts[a["id"]] if a["kind"] == "corpus" else queries[a["id"]]["correct_text"]
                assert text[a["start"]:a["end"]] == a["text"]
                assert normalize(a["text"]) == f["key"]
    for q in dev:
        for mention in q["correct_entities"]:
            assert (normalize(mention["text"]), mention["type"]) in forms_by_type
        for typo in q["typos"]:
            assert normalize(typo["wrong"]) not in ownership, "Dev typo added as valid entity form"
    assert normalize("Королёв") == normalize("КОРОЛЕВ")
    assert [m.span() for m in exact_mentions("«МГУ», МГУшник и МГУ-2", "МГУ")] == [(1, 4)]
    assert [m.span() for m in exact_mentions("О Королёве.", "Королеве")] == [(2, 10)]
    for path, expected_hash in metadata["input_sha256"].items():
        assert digest(ROOT / path) == expected_hash
    # Dev-only diagnostics do not change the corpus or dictionaries.
    correct_tokens = [t.lower().replace("ё", "е") for q in dev
                      for t in re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*", q["correct_text"])]
    unknown = Counter(t for t in correct_tokens if t not in words)
    report = dict(status="passed", corpus_tokens=sum(words.values()), wordforms=len(words),
        entities=len(entities), entity_forms=metadata["entity_forms"],
        checks=["independent frequency recount", "display spellings and document totals",
                "original-source provenance offsets", "all correct dev entities covered",
                "dev typo exclusion", "word-boundary and ё/е matching", "input hashes",
                "deterministic rebuild with test-containing files blocked"],
        builder_read_files=sorted(accessed), test_used=False,
        dev_correct_token_coverage=dict(known=sum(t in words for t in correct_tokens), total=len(correct_tokens),
                                       missing_wordforms=dict(sorted(unknown.items()))),
        note="Coverage is a dictionary diagnostic on dev, not correction accuracy or test evaluation.")
    (directory / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
