"""Build dictionaries from the corpus and dev only, using stdlib Python."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from entity_seeds import DEV_LINKS, catalogue
from text_utils import exact_mentions, normalize, tokens

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    manifest = read_json(ROOT / "data/corpus/manifest.json")
    dev = read_json(ROOT / "data/queries/queries_dev.json")
    assert all(q["split"] == "dev" for q in dev)
    counts = Counter()
    variants = defaultdict(Counter)
    document_counts = defaultdict(Counter)
    texts = {}
    input_hashes = {}
    for doc in manifest:
        path = ROOT / doc["text_file"]
        text = path.read_text()
        assert digest(path) == doc["sha256"], f"Corpus changed: {doc['id']}"
        texts[doc["id"]] = text
        input_hashes[doc["text_file"]] = doc["sha256"]
        for match in tokens(text):
            form = match.group()
            key = normalize(form)
            counts[key] += 1
            variants[key][form] += 1
            document_counts[key][doc["id"]] += 1
    word_dictionary = dict(sorted(counts.items()))
    word_details = {}
    for key in word_dictionary:
        display = sorted(variants[key], key=lambda x: (-variants[key][x], x != x.lower(), x))[0]
        word_details[key] = dict(frequency=counts[key], display=display,
                                observed_spellings=dict(sorted(variants[key].items())),
                                document_frequencies=dict(sorted(document_counts[key].items())))
    entries = catalogue()
    by_id = {e["id"]: e for e in entries}
    dev_mentions = defaultdict(list)
    for query in dev:
        links = DEV_LINKS[query["id"]]
        assert len(links) == len(query["correct_entities"])
        for entity_id, mention in zip(links, query["correct_entities"]):
            entry = by_id[entity_id]
            assert entry["type"] == mention["type"]
            assert query["correct_text"][mention["start"]:mention["end"]] == mention["text"]
            entry["forms"].append(mention["text"])
            dev_mentions[entity_id].append((query["id"], mention))
    entities = []
    provenance = {}
    for entry in entries:
        seen = set()
        forms = []
        for form in entry["forms"]:
            key = normalize(form)
            if key not in seen:
                seen.add(key)
                forms.append(form)
        base = entry["base_source"]
        entity_evidence = []
        if base["kind"] == "corpus":
            for form in forms:
                for match in exact_mentions(texts[base["id"]], form):
                    entity_evidence.append(dict(document_id=base["id"], start=match.start(),
                                                end=match.end(), text=texts[base["id"]][match.start():match.end()]))
            assert entity_evidence, f"No corpus evidence for {entry['id']}"
        else:
            assert any(qid == base["id"] for qid, _ in dev_mentions[entry["id"]])
        form_sources = []
        for form in forms:
            attestations = []
            for doc_id, text in texts.items():
                for match in exact_mentions(text, form):
                    attestations.append(dict(kind="corpus", id=doc_id, start=match.start(),
                                             end=match.end(), text=text[match.start():match.end()]))
            for qid, mention in dev_mentions[entry["id"]]:
                if normalize(form) == normalize(mention["text"]):
                    attestations.append(dict(kind="dev", id=qid, text_field="correct_text",
                                             start=mention["start"], end=mention["end"], text=mention["text"]))
            form_sources.append(dict(form=form, key=normalize(form),
                method="attested" if attestations else "manual_form",
                base_source=base, attestations=attestations))
        entities.append(dict(id=entry["id"], name=entry["name"], type=entry["type"], forms=forms))
        provenance[entry["id"]] = dict(base_source=base, base_evidence=entity_evidence, forms=form_sources)
    index = defaultdict(list)
    for entry in entities:
        for form in entry["forms"]:
            index[normalize(form)].append(entry["id"])
    collisions = {k: v for k, v in index.items() if len(v) > 1}
    for name in ("data/corpus/manifest.json", "data/queries/queries_dev.json", "data/normalization.json",
                 "entity_seeds.py", "text_utils.py", "build_dictionaries.py"):
        input_hashes[name] = digest(ROOT / name)
    metadata = dict(corpus_documents=len(manifest), corpus_tokens=sum(counts.values()),
        unique_wordforms=len(counts), entities=len(entities),
        entities_by_type=dict(Counter(e["type"] for e in entities)),
        entity_forms=sum(len(e["forms"]) for e in entities),
        attested_forms=sum(f["method"] == "attested" for p in provenance.values() for f in p["forms"]),
        manual_forms=sum(f["method"] == "manual_form" for p in provenance.values() for f in p["forms"]),
        normalized_form_collisions=collisions, input_sha256=input_hashes,
        word_sources="corpus only", entity_sources="selected corpus topics + correct dev entity labels + manually enumerated forms",
        scope="24 principal corpus entities plus 3 dev-only entities; incidental names are not exhaustively covered",
        test_used=False, algorithms_evaluated=False)
    return word_dictionary, word_details, entities, provenance, metadata


def main():
    values = build()
    output = ROOT / "data/dictionaries"
    output.mkdir(exist_ok=True)
    names = ["word_dictionary.json", "word_details.json", "entity_dictionary.json", "entity_provenance.json", "build_metadata.json"]
    for name, value in zip(names, values):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    words, _, entities, _, meta = values
    lines = ["# Подготовленные словари", "", f"Словоформ: {len(words)}; токенов корпуса: {meta['corpus_tokens']}; сущностей: {len(entities)}; допустимых форм: {meta['entity_forms']}.", "",
             "## Частые словоформы", "", "| Ключ | Частота |", "|---|---:|"]
    for key, count in sorted(words.items(), key=lambda x: (-x[1], x[0]))[:25]:
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Справочник сущностей", "", "Полный перечень вариантов и происхождение каждой формы находятся в JSON. Ниже приведены примеры.", "", "| ID | Название | Тип | Примеры форм |", "|---|---|---|---|"])
    for e in entities:
        examples = "; ".join(e["forms"][:3])
        lines.append(f"| {e['id']} | {e['name']} | {e['type']} | {examples} |")
    (ROOT / "dictionaries_preview.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "input_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
