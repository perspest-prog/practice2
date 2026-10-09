"""Validate the dataset, not algorithm quality. Uses only the standard library."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOKEN = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")


def load(path):
    return json.loads((ROOT / path).read_text())


def verify_operation(correct, wrong, operation):
    if operation == "deletion":
        return any(correct[:i] + correct[i + 1:] == wrong for i in range(len(correct)))
    if operation == "insertion":
        return any(wrong[:i] + wrong[i + 1:] == correct for i in range(len(wrong)))
    if operation == "substitution":
        return len(correct) == len(wrong) and sum(a != b for a, b in zip(correct, wrong)) == 1
    if operation == "transposition":
        return any(correct[i] != correct[i + 1] and
                   correct[:i] + correct[i + 1] + correct[i] + correct[i + 2:] == wrong
                   for i in range(len(correct) - 1))
    return False


def main():
    manifest = load("data/corpus/manifest.json")
    assert len(manifest) == 24
    assert Counter(d["category"] for d in manifest) == {"person": 8, "organization": 8, "location": 8}
    assert len({d["page_id"] for d in manifest}) == 24
    assert len({d["sha256"] for d in manifest}) == 24
    texts = {}
    shares = []
    for d in manifest:
        text = (ROOT / d["text_file"]).read_text()
        assert len(text) == d["chars"] >= 200
        assert hashlib.sha256(text.encode()).hexdigest() == d["sha256"]
        ru, latin = len(re.findall(r"[А-Яа-яЁё]", text)), len(re.findall(r"[A-Za-z]", text))
        shares.append(ru / (ru + latin))
        assert shares[-1] > .85
        assert not re.search(r"<[^>]+>|\{\{|\}\}|\[\.\.\.\]|\[…\]|МФА:", text)
        snapshot = load(d["raw_file"])
        assert snapshot["response"]["query"]["pages"][0]["revisions"][0]["revid"] == d["revision_id"]
        texts[d["id"]] = text
    evidence = []
    for doc_id, mention, kind in (("person_01", "Михаил Васильевич Ломоносов", "PERSON"),
                                  ("organization_01", "Московский государственный университет имени М. В. Ломоносова", "ORG"),
                                  ("location_05", "Нижний Новгород", "LOC")):
        start = texts[doc_id].index(mention)
        evidence.append(dict(document_id=doc_id, start=start, end=start + len(mention), text=mention, type=kind))
    all_records = []
    summary = {}
    for split in ("dev", "test"):
        records = load(f"data/queries/queries_{split}.json")
        assert len(records) == 20
        groups = Counter(r["group"] for r in records)
        assert groups == {"clean": 5, "ordinary_typo": 5, "entity_typo": 5, "rare_correct": 5}
        types, operations = Counter(), Counter()
        multiword = 0
        for r in records:
            assert r["split"] == split
            for field, text_key in (("original_entities", "original_text"), ("correct_entities", "correct_text")):
                text = r[text_key]
                last_end = 0
                for e in r[field]:
                    assert 0 <= e["start"] < e["end"] <= len(text)
                    assert e["start"] >= last_end
                    assert text[e["start"]:e["end"]] == e["text"]
                    assert e["type"] in {"PERSON", "ORG", "LOC"}
                    assert e["start"] == 0 or not text[e["start"] - 1].isalpha()
                    assert e["end"] == len(text) or not text[e["end"]].isalpha()
                    assert e["text"] == e["text"].strip(' «».,!?')
                    last_end = e["end"]
            types.update(e["type"] for e in r["original_entities"])
            multiword += sum(" " in e["text"] for e in r["correct_entities"])
            token_spans = {(m.start(), m.end()) for m in TOKEN.finditer(r["original_text"])}
            rebuilt = r["original_text"]
            for t in reversed(r["typos"]):
                assert (t["start"], t["end"]) in token_spans
                assert rebuilt[t["start"]:t["end"]] == t["wrong"]
                assert verify_operation(t["correct"], t["wrong"], t["operation"])
                intersections = [e for e in r["original_entities"] if t["start"] < e["end"] and e["start"] < t["end"]]
                assert bool(intersections) == (r["group"] == "entity_typo")
                if intersections:
                    assert any(e["start"] <= t["start"] and t["end"] <= e["end"] for e in intersections)
                rebuilt = rebuilt[:t["start"]] + t["correct"] + rebuilt[t["end"]:]
                operations[t["operation"]] += 1
            assert rebuilt == r["correct_text"]
            assert bool(r["typos"]) == (r["group"] in {"ordinary_typo", "entity_typo"})
        assert set(types) == {"PERSON", "ORG", "LOC"}
        assert set(operations) == {"deletion", "insertion", "substitution", "transposition"}
        assert multiword >= 3
        summary[split] = dict(queries=len(records), groups=dict(groups), entities_by_type=dict(types),
                              typo_operations=dict(operations), multiword_mentions=multiword,
                              rare_cases=dict(Counter(r["rare_case"] for r in records if r["group"] == "rare_correct")),
                              no_entity_queries=sum(not r["original_entities"] for r in records))
        all_records.extend(records)
    for key in ("id", "family_id", "original_text", "correct_text"):
        assert len({r[key] for r in all_records}) == 40, f"Duplicate {key}"
    corpus = "\n".join(texts.values()).lower().replace("ё", "е")
    outside = ["чижевск", "буслаев", "мезень"]
    absent = [stem for stem in outside if stem not in corpus]
    assert len(absent) == 3
    report = dict(status="passed", checks="corpus hashes, sources, language diagnostic, duplicates, groups, gold spans, whole-token typos, operation types, reconstruction, entity overlaps, split families",
                  corpus=dict(documents=24, chars=sum(d["chars"] for d in manifest),
                              words=sum(d["words"] for d in manifest), min_russian_letter_share=min(shares),
                              entity_presence_examples=evidence), queries=summary,
                  outside_test_name_stems_absent_from_corpus=absent,
                  evaluation_run=False, note="Test gold was read only for structural validation, never for dictionary construction or parameter tuning.")
    (ROOT / "data/validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Запросы лабораторной 4", "", "Разметка задана вручную, позиции рассчитаны по выбранным фрагментам и проверены валидатором. Метрики алгоритмов ещё не вычислялись.", ""]
    for split in ("dev", "test"):
        lines += [f"## {split}: 20 запросов", "", "| ID | Группа | Исходный запрос | Правильный запрос | Сущности в правильном тексте |", "|---|---|---|---|---|"]
        for r in [x for x in all_records if x["split"] == split]:
            entities = "; ".join(f"{e['text']} ({e['type']})" for e in r["correct_entities"]) or "—"
            lines.append(f"| {r['id']} | {r['group']} | {r['original_text']} | {r['correct_text']} | {entities} |")
        lines.append("")
    (ROOT / "queries_preview.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
