"""Dictionary/rules recognizer and pretrained Russian Slovnet news NER."""
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

from text_utils import exact_mentions

ROOT = Path(__file__).resolve().parent


class ModelLoadError(RuntimeError):
    pass


def non_overlapping_longest(candidates):
    """Earliest position first, longest at the same position; deterministic ties."""
    candidates.sort(key=lambda e: (e["start"], -(e["end"] - e["start"]),
                                   e["method"] != "dictionary", e["type"], e.get("entity_id", "")))
    selected = []
    cursor = 0
    for entity in candidates:
        if entity["start"] >= cursor:
            selected.append(entity)
            cursor = entity["end"]
    return selected


class DictionaryNER:
    # One-word city name (including hyphens) explicitly introduced by 'город'.
    CITY = re.compile(r"(?<!\w)(?i:город|города|городу|городом|городе)\s+"
                      r"(?P<name>[А-ЯЁ][а-яё]+(?:-[А-ЯЁ]?[а-яё]+)*)(?![\w-])")

    def __init__(self, entries, context_rules=True):
        if not isinstance(entries, list):
            raise ValueError("Справочник сущностей должен быть массивом")
        for entry in entries:
            if entry.get("type") not in {"PERSON", "ORG", "LOC"} or not entry.get("forms"):
                raise ValueError("Некорректная запись справочника сущностей")
        self.entries = entries
        self.context_rules = context_rules

    @classmethod
    def from_file(cls, path, **kwargs):
        obj = cls(json.loads(Path(path).read_text()), **kwargs)
        obj.source_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return obj

    def recognize(self, text):
        if not isinstance(text, str):
            raise ValueError("Запрос должен быть строкой")
        candidates = []
        for entry in self.entries:
            for form in entry["forms"]:
                for match in exact_mentions(text, form):
                    candidates.append(dict(start=match.start(), end=match.end(),
                        text=text[match.start():match.end()], type=entry["type"],
                        entity_id=entry["id"], method="dictionary"))
        if self.context_rules:
            for match in self.CITY.finditer(text):
                a, b = match.span("name")
                candidates.append(dict(start=a, end=b, text=text[a:b], type="LOC", method="city_context_rule"))
        return non_overlapping_longest(candidates)

    def metadata(self):
        return dict(name="dictionary_and_rules", context_rules=self.context_rules,
                    source_sha256=getattr(self, "source_sha256", None),
                    rule="город/города/городу/городом/городе + capitalized one-word city name, including hyphens",
                    limitation="Does not handle unknown multiword names or lowercase city names; may classify a capitalized common word or a typo as LOC.")


class SlovnetNER:
    LABELS = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}

    def __init__(self, model_dir=ROOT / "models"):
        directory = Path(model_dir)
        try:
            from navec import Navec
            from slovnet import NER
            packs = [directory / "slovnet_ner_news_v1.tar", directory / "navec_news_v1_1B_250K_300d_100q.tar"]
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in packs}
            manifest = json.loads((directory / "manifest.json").read_text())
            if any(hashes[name] != manifest[name]["sha256"] for name in hashes):
                raise ValueError("Хеши модели не совпадают с сохранённым манифестом")
            self.model = NER.load(str(packs[0]))
            self.model.navec(Navec.load(str(packs[1])))
            self.info = dict(name="slovnet_ner_news_v1", embedding="navec_news_v1_1B_250K_300d_100q",
                source="https://github.com/natasha/slovnet", sha256=hashes,
                packages={name: importlib.metadata.version(name) for name in ("slovnet", "navec", "numpy", "razdel")},
                label_mapping=self.LABELS, ignored_labels="Any label not in PER/ORG/LOC is ignored; O is not an entity.",
                execution="Offline CPU inference, pretrained weights unchanged")
        except Exception as exc:
            raise ModelLoadError("Не удалось загрузить Slovnet NER. Установите lab4/requirements.txt "
                                 "в lab4/.venv и запустите download_models.py. Причина: " + str(exc)) from exc

    def recognize(self, text):
        if not isinstance(text, str):
            raise ValueError("Запрос должен быть строкой")
        if not text.strip():
            return []
        markup = self.model(text)
        result = []
        for span in markup.spans:
            kind = self.LABELS.get(span.type)
            if kind:
                result.append(dict(start=span.start, end=span.stop, text=text[span.start:span.stop],
                                   type=kind, model_label=span.type, method="pretrained_model"))
        return result

    def metadata(self):
        return self.info


def create_recognizer(name, dictionary_dir, model_dir):
    if name == "dictionary":
        return DictionaryNER.from_file(Path(dictionary_dir) / "entity_dictionary.json")
    if name == "model":
        return SlovnetNER(model_dir)
    raise ValueError("Неизвестный распознаватель")
