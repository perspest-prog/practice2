"""Run two original NER baselines and four pipelines on dev, without scoring."""
import hashlib
import json
import platform
from pathlib import Path

from ner import create_recognizer
from pipeline import process
from spelling import SpellCorrector

ROOT = Path(__file__).resolve().parent


def validate_entities(text, entities):
    previous = 0
    for entity in entities:
        a, b = entity["start"], entity["end"]
        assert previous <= a < b <= len(text)
        assert text[a:b] == entity["text"]
        assert entity["type"] in {"PERSON", "ORG", "LOC"}
        previous = b


def main():
    raw = json.loads((ROOT / "data/queries/queries_dev.json").read_text())
    # Use original inputs only; no evaluation against dev gold at this stage.
    queries = [(r["id"], r["original_text"]) for r in raw]
    output_dir = ROOT / "output/ner"
    output_dir.mkdir(parents=True, exist_ok=True)
    corrector = SpellCorrector.from_files(ROOT / "data/dictionaries")
    saved = {}
    for name in ("dictionary", "model"):
        recognizer = create_recognizer(name, ROOT / "data/dictionaries", ROOT / "models")
        baseline = [dict(id=qid, original_text=text, entities=recognizer.recognize(text),
                         entity_positions="original_text") for qid, text in queries]
        for row in baseline:
            validate_entities(row["original_text"], row["entities"])
        saved[f"{name}_original"] = baseline
        for order in ("A", "B"):
            rows = [dict(id=qid, **process(text, corrector, recognizer, order)) for qid, text in queries]
            for row in rows:
                validate_entities(row["original_text"], row["initial_entities"])
                validate_entities(row["proposed_text"], row["entities"])
                for c in row["changes"]:
                    assert row["original_text"][c["start"]:c["end"]] == c["old"]
                    assert row["proposed_text"][c["proposed_start"]:c["proposed_end"]] == c["new"]
            saved[f"{name}_{order}"] = rows
        for key, rows in saved.items():
            if key.startswith(name):
                config = dict(recognizer=recognizer.metadata(), threshold=1, use_frequencies=True)
                if not key.endswith("original"):
                    config.update(order=key[-1], protect_entities=key.endswith("B"))
                payload = dict(stage="dev_demonstration", evaluation_run=False, test_used=False,
                               config=config, results=rows)
                (output_dir / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Распознавание и порядки A/B на dev", "", "Демонстрация при пороге 1 и частотах; без итоговой оценки и без запуска test.", ""]
    for name in ("dictionary", "model"):
        lines += [f"## {name}", "", "| ID | Исходный запрос | Сущности без коррекции | Порядок A | Сущности A | Порядок B | Сущности B |",
                  "|---|---|---|---|---|---|---|"]
        for baseline, a, b in zip(saved[f"{name}_original"], saved[f"{name}_A"], saved[f"{name}_B"]):
            def show(row):
                return "; ".join(f"{e['text']} ({e['type']}, [{e['start']}, {e['end']}))" for e in row["entities"]) or "—"
            lines.append(f"| {baseline['id']} | {baseline['original_text']} | {show(baseline)} | {a['proposed_text']} | {show(a)} | {b['proposed_text']} | {show(b)} |")
        lines.append("")
    (ROOT / "ner_preview.md").write_text("\n".join(lines) + "\n")
    sources = [ROOT / file for file in ("ner.py", "pipeline.py", "spelling.py", "text_utils.py", "demo_ner.py",
               "data/queries/queries_dev.json", "data/dictionaries/word_dictionary.json",
               "data/dictionaries/word_details.json", "data/dictionaries/entity_dictionary.json", "models/manifest.json")]
    metadata = dict(status="passed", queries=20, configurations=6, checked_results=120,
                    checks="entity spans/non-overlap and replacement source/proposed ranges",
                    source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                    python=platform.python_version(), model=create_recognizer("model", ROOT / "data/dictionaries", ROOT / "models").metadata(),
                    test_used=False, evaluation_run=False)
    (output_dir / "validation.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("Saved two original baselines + four A/B combinations; verified 120 dev results")


if __name__ == "__main__":
    main()
