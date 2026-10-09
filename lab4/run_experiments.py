"""Separate dev tuning and one-shot final evaluation under a frozen config."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from evaluation import evaluate
from ner import create_recognizer
from pipeline import process
from spelling import SpellCorrector

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/evaluation"
LOCK = OUT / "frozen_config.json"
SELECTION_RULE = "Common settings for four pipelines: minimize mean dev false-change rate, then maximize mean exact typo correction rate, then mean entity F1; ties prefer threshold 1 and frequencies."


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def hashes():
    paths = [ROOT / name for name in ("evaluation.py", "run_experiments.py", "ner.py", "pipeline.py",
             "spelling.py", "text_utils.py", "requirements.txt", "models/manifest.json", "data/normalization.json",
             "data/dictionaries/word_dictionary.json", "data/dictionaries/word_details.json",
             "data/dictionaries/entity_dictionary.json", "data/corpus/manifest.json", "data/queries/queries_dev.json")]
    manifest = load(ROOT / "data/corpus/manifest.json")
    paths.extend(ROOT / d["text_file"] for d in manifest)
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def name_for(name, order, threshold, frequencies, protection):
    return f"{name}_{order}_d{threshold}_freq{int(frequencies)}_protect{int(protection)}"


def experiment(gold, recognizer, name, order=None, threshold=1, frequencies=True, protection=True):
    if order is None:
        rows = [dict(id=g["id"], original_text=g["original_text"], entities=recognizer.recognize(g["original_text"]),
                     entity_positions="original_text") for g in gold]
        key = f"{name}_original"
        config = dict(recognizer=name, order=None)
    else:
        protection = order == "B" and protection
        corrector = SpellCorrector.from_files(ROOT / "data/dictionaries", threshold=threshold, use_frequencies=frequencies)
        rows = [dict(id=g["id"], **process(g["original_text"], corrector, recognizer, order, protection)) for g in gold]
        key = name_for(name, order, threshold, frequencies, protection)
        config = dict(recognizer=name, order=order, threshold=threshold, use_frequencies=frequencies, protection=protection)
    return key, dict(config=config, recognizer_metadata=recognizer.metadata(), results=rows,
                     metrics=evaluate(gold, rows, correction=order is not None))


def compact(payload):
    return dict(config=payload["config"], metrics={k: v for k, v in payload["metrics"].items() if k != "query_details"})


def table(path, runs, heading):
    lines = [heading, "", "Доли указаны в процентах; P/R/F1 по точному совпадению перенесённых границ и типа.", "",
             "| Режим | Исправлено опечаток | Повреждено правильных слов | Precision | Recall | F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    csv_rows = []
    for key, item in runs.items():
        m = item["metrics"]
        s, n = m["spelling"], m["ner"]
        correction = f"{100*s['correction_rate']:.1f}% ({s['corrected_typos']}/{s['typo_tokens']})" if s else "—"
        damaged = f"{100*s['false_change_rate']:.1f}% ({s['damaged_correct_tokens']}/{s['correct_tokens']})" if s else "—"
        lines.append(f"| {key} | {correction} | {damaged} | {100*n['precision']:.1f}% | {100*n['recall']:.1f}% | {100*n['f1']:.1f}% |")
        csv_rows.append(dict(mode=key, corrected_typos=s["corrected_typos"] if s else "", typo_tokens=s["typo_tokens"] if s else "",
            correction_rate=s["correction_rate"] if s else "", damaged_correct_tokens=s["damaged_correct_tokens"] if s else "",
            correct_tokens=s["correct_tokens"] if s else "", false_change_rate=s["false_change_rate"] if s else "",
            precision=n["precision"], recall=n["recall"], f1=n["f1"], tp=n["tp"], fp=n["fp"], fn=n["fn"]))
    lines += ["", "## Метрики по типам", "", "| Режим | Тип | TP | FP | FN | Precision | Recall | F1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, item in runs.items():
        for kind, n in item["metrics"]["ner_by_type"].items():
            lines.append(f"| {key} | {kind} | {n['tp']} | {n['fp']} | {n['fn']} | {n['precision']:.3f} | {n['recall']:.3f} | {n['f1']:.3f} |")
    path.write_text("\n".join(lines) + "\n")
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)


def dev():
    if LOCK.exists() or (OUT / "test_summary.json").exists():
        raise ValueError("Конфигурация уже зафиксирована; повторная настройка в этом запуске запрещена")
    gold = load(ROOT / "data/queries/queries_dev.json")
    runs = {}
    for name in ("dictionary", "model"):
        recognizer = create_recognizer(name, ROOT / "data/dictionaries", ROOT / "models")
        key, run = experiment(gold, recognizer, name)
        runs[key] = run
        for threshold in (1, 2):
            for frequencies in (True, False):
                for order in ("A", "B"):
                    for protection in ((False,) if order == "A" else (True, False)):
                        key, run = experiment(gold, recognizer, name, order, threshold, frequencies, protection)
                        runs[key] = run
    for key, run in runs.items():
        save(OUT / "dev" / f"{key}.json", run)
    comparisons = []
    for name in ("dictionary", "model"):
        for order in ("A", "B"):
            protected = order == "B"
            pairs = [("threshold", name_for(name, order, 1, True, protected), name_for(name, order, 2, True, protected)),
                     ("frequencies", name_for(name, order, 1, True, protected), name_for(name, order, 1, False, protected))]
            if order == "B":
                pairs.append(("protection", name_for(name, order, 1, True, True), name_for(name, order, 1, True, False)))
            for factor, before, after in pairs:
                a, b = runs[before]["metrics"], runs[after]["metrics"]
                comparisons.append(dict(factor=factor, before=before, after=after,
                    corrected_typos_delta=b["spelling"]["corrected_typos"]-a["spelling"]["corrected_typos"],
                    damaged_tokens_delta=b["spelling"]["damaged_correct_tokens"]-a["spelling"]["damaged_correct_tokens"],
                    ner_f1_delta=b["ner"]["f1"]-a["ner"]["f1"]))
    candidates = []
    for threshold in (1, 2):
        for frequencies in (True, False):
            four = [runs[name_for(name, order, threshold, frequencies, order == "B")]["metrics"]
                    for name in ("dictionary", "model") for order in ("A", "B")]
            candidates.append(dict(threshold=threshold, use_frequencies=frequencies,
                mean_false_change_rate=sum(m["spelling"]["false_change_rate"] for m in four)/4,
                mean_correction_rate=sum(m["spelling"]["correction_rate"] for m in four)/4,
                mean_entity_f1=sum(m["ner"]["f1"] for m in four)/4))
    chosen = max(candidates, key=lambda c: (-c["mean_false_change_rate"], c["mean_correction_rate"],
                   c["mean_entity_f1"], -c["threshold"], c["use_frequencies"]))
    summary = dict(split="dev", queries=20, runs={k: compact(v) for k, v in runs.items()},
                   one_factor_comparisons=comparisons, selection_rule=SELECTION_RULE, candidates=candidates, chosen=chosen)
    save(OUT / "dev_summary.json", summary)
    table(OUT / "dev_comparison.md", runs, "# Эксперименты на наборе настройки")
    lock = dict(created_at=datetime.now(timezone.utc).isoformat(), selected=chosen,
                selection_rule=SELECTION_RULE, source_sha256=hashes(),
                dev_summary_sha256=hashlib.sha256((OUT / "dev_summary.json").read_bytes()).hexdigest(),
                test_used_for_selection=False)
    save(LOCK, lock)
    print(json.dumps(chosen, ensure_ascii=False, indent=2))


def final():
    marker = OUT / "test_started.json"
    if marker.exists():
        raise ValueError("Итоговый запуск уже начат или выполнен; повторный запуск test запрещён")
    lock = load(LOCK)
    if hashes() != lock["source_sha256"]:
        raise ValueError("Код, корпус или словари изменились после фиксации конфигурации")
    if hashlib.sha256((OUT / "dev_summary.json").read_bytes()).hexdigest() != lock["dev_summary_sha256"]:
        raise ValueError("Изменились сохранённые результаты настройки")
    config = lock["selected"]
    save(marker, dict(started_at=datetime.now(timezone.utc).isoformat(), frozen_config_sha256=hashlib.sha256(LOCK.read_bytes()).hexdigest()))
    gold = load(ROOT / "data/queries/queries_test.json")
    runs = {}
    for name in ("dictionary", "model"):
        recognizer = create_recognizer(name, ROOT / "data/dictionaries", ROOT / "models")
        key, run = experiment(gold, recognizer, name)
        runs[key] = run
        for order in ("A", "B"):
            key, run = experiment(gold, recognizer, name, order, config["threshold"], config["use_frequencies"], True)
            runs[key] = run
    for key, run in runs.items():
        save(OUT / "test" / f"{key}.json", run)
    summary = dict(split="test", queries=20, frozen_config_sha256=hashlib.sha256(LOCK.read_bytes()).hexdigest(),
                   selected=config, test_sha256=hashlib.sha256((ROOT / "data/queries/queries_test.json").read_bytes()).hexdigest(),
                   runs={k: compact(v) for k, v in runs.items()},
                   transfer_failures=sum(r["metrics"]["transfer_failure_count"] for r in runs.values()))
    save(OUT / "test_summary.json", summary)
    table(OUT / "test_comparison.md", runs, "# Итоговая проверка на отложенных запросах")
    print("Final test: two original baselines + four frozen A/B pipelines saved; transfer failures:", summary["transfer_failures"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["dev", "final"])
    args = parser.parse_args()
    if args.stage == "dev":
        normal_open = Path.open

        def forbid_test(path, *a, **kw):
            if "queries_test" in str(path) or path.name == "prepare_queries.py":
                raise ValueError("Проверочные данные запрещены на этапе настройки")
            return normal_open(path, *a, **kw)

        with patch.object(Path, "open", forbid_test):
            dev()
    else:
        final()


if __name__ == "__main__":
    main()
