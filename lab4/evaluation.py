"""Gold scoring with source-to-output span transfer, micro entity metrics."""
from collections import Counter

from spelling import transfer_span
from text_utils import normalize, tokens

TYPES = ("PERSON", "ORG", "LOC")


def entity_metrics(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def score_query(gold, result, correction=True):
    source = gold["original_text"]
    if result["original_text"] != source:
        raise ValueError("Исходный текст результата не совпадает с эталоном")
    output = result["proposed_text"] if correction else source
    changes = result["changes"] if correction else []
    # Verify actual edits before using the log as coordinate mapping.
    rebuilt = source
    for c in reversed(changes):
        if rebuilt[c["start"]:c["end"]] != c["old"]:
            raise ValueError("Журнал замен не соответствует исходному тексту")
        rebuilt = rebuilt[:c["start"]] + c["new"] + rebuilt[c["end"]:]
    if rebuilt != output:
        raise ValueError("Журнал замен не восстанавливает предложенный текст")
    mapped, failures = [], []
    for entity in gold["original_entities"]:
        if source[entity["start"]:entity["end"]] != entity["text"]:
            raise ValueError("Некорректная исходная эталонная сущность")
        try:
            a, b = transfer_span(entity["start"], entity["end"], changes)
            mapped.append(dict(start=a, end=b, type=entity["type"], text=output[a:b]))
        except ValueError as exc:
            failures.append(dict(entity=entity, reason=str(exc), counted_as="FN"))
    expected = Counter((e["start"], e["end"], e["type"]) for e in mapped)
    predicted = Counter()
    for entity in result["entities"]:
        a, b, kind = entity["start"], entity["end"], entity["type"]
        if kind not in TYPES or not 0 <= a < b <= len(output) or output[a:b] != entity["text"]:
            raise ValueError("Некорректные границы, текст или тип предсказанной сущности")
        predicted[(a, b, kind)] += 1
    per_type = {}
    for kind in TYPES:
        wanted = Counter({k: v for k, v in expected.items() if k[2] == kind})
        found = Counter({k: v for k, v in predicted.items() if k[2] == kind})
        per_type[kind] = dict(tp=sum((wanted & found).values()), fp=sum((found - wanted).values()),
                              fn=sum((wanted - found).values()) + sum(f["entity"]["type"] == kind for f in failures))
    spelling = None
    if correction:
        typo_spans = {(t["start"], t["end"]): t for t in gold["typos"]}
        edits = {(c["start"], c["end"]): c for c in changes}
        source_tokens = list(tokens(source.replace("\u2010", "-").replace("\u2011", "-")))
        if not set(typo_spans) <= {m.span() for m in source_tokens}:
            raise ValueError("Эталонная опечатка должна занимать целый токен")
        if not set(edits) <= {m.span() for m in source_tokens}:
            raise ValueError("Замена должна занимать целый токен")
        token_results = {(t["start"], t["end"]): t for t in result["tokens"]}
        corrected, damaged, top3 = 0, 0, 0
        for match in source_tokens:
            span = match.span()
            original = source[match.start():match.end()]
            proposed = edits[span]["new"] if span in edits else original
            if span in typo_spans:
                target = normalize(typo_spans[span]["correct"])
                corrected += span in edits and normalize(proposed) == target
                candidates = token_results.get(span, {}).get("candidates", [])
                top3 += any(normalize(c["text"]) == target for c in candidates)
            else:
                damaged += normalize(original) != normalize(proposed)
        spelling = dict(corrected_typos=corrected, typo_tokens=len(typo_spans),
                        damaged_correct_tokens=damaged, correct_tokens=len(source_tokens) - len(typo_spans),
                        top3_hits=top3)
    return dict(id=gold["id"], group=gold["group"], spelling_counts=spelling,
                entity_counts_by_type=per_type, transferred_gold_entities=mapped,
                transfer_failures=failures)


def aggregate(details, correction):
    per_type = {}
    for kind in TYPES:
        counts = {key: sum(r["entity_counts_by_type"][kind][key] for r in details) for key in ("tp", "fp", "fn")}
        per_type[kind] = entity_metrics(counts)
    total = {key: sum(m[key] for m in per_type.values()) for key in ("tp", "fp", "fn")}
    spelling = None
    if correction:
        counts = {key: sum(r["spelling_counts"][key] for r in details)
                  for key in ("corrected_typos", "typo_tokens", "damaged_correct_tokens", "correct_tokens", "top3_hits")}
        spelling = dict(**counts,
            correction_rate=counts["corrected_typos"] / counts["typo_tokens"] if counts["typo_tokens"] else 0.0,
            false_change_rate=counts["damaged_correct_tokens"] / counts["correct_tokens"] if counts["correct_tokens"] else 0.0,
            top3_rate=counts["top3_hits"] / counts["typo_tokens"] if counts["typo_tokens"] else 0.0)
    return dict(queries=len(details), spelling=spelling, ner=entity_metrics(total), ner_by_type=per_type,
                transfer_failure_count=sum(len(r["transfer_failures"]) for r in details))


def evaluate(gold, results, correction=True):
    by_id = {r["id"]: r for r in results}
    if len(by_id) != len(results) or set(by_id) != {r["id"] for r in gold}:
        raise ValueError("Идентификаторы результатов должны точно совпадать с эталоном")
    details = [score_query(g, by_id[g["id"]], correction) for g in gold]
    return dict(**aggregate(details, correction),
                by_group={group: aggregate([r for r in details if r["group"] == group], correction)
                          for group in sorted({r["group"] for r in details})},
                query_details=details, zero_denominator_value=0,
                entity_match="exact transferred original-gold boundaries and type; spelling scored separately")
