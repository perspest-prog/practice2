"""Serialize manually authored queries and labels; no recognition model is used."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Fields: group, family, correct text, manually selected (text, type) mentions,
# optional (wrong token, correct token, operation in correct -> original direction).
DEV = [
    ("clean", "lomonosov_biography", "Биография Михаила Ломоносова.", [("Михаила Ломоносова", "PERSON")]),
    ("clean", "msu_location", "Где находится Московский государственный университет?", [("Московский государственный университет", "ORG")]),
    ("clean", "nizhny_history", "История Нижнего Новгорода.", [("Нижнего Новгорода", "LOC")]),
    ("clean", "gagarin_moscow", "Связь Юрия Гагарина с Москвой.", [("Юрия Гагарина", "PERSON"), ("Москвой", "LOC")]),
    ("clean", "education_structure", "Как устроено высшее образование?", []),
    ("ordinary_typo", "mendeleev_biography", "Биография Дмитрия Менделеева.", [("Дмитрия Менделеева", "PERSON")], ("Биогрфия", "Биография", "deletion")),
    ("ordinary_typo", "ras_history", "История Российской академии наук.", [("Российской академии наук", "ORG")], ("Исторрия", "История", "insertion")),
    ("ordinary_typo", "korolev_discoveries", "Научные открытия Сергея Королёва.", [("Сергея Королёва", "PERSON")], ("открвтия", "открытия", "substitution")),
    ("ordinary_typo", "lobachevsky_study", "Где учился Николай Лобачевский?", [("Николай Лобачевский", "PERSON")], ("учлися", "учился", "transposition")),
    ("ordinary_typo", "kazan_education", "Расскажи об образовании в Казани.", [("Казани", "LOC")], ("образовнии", "образовании", "deletion")),
    ("entity_typo", "moscow_science", "Наука в Москве.", [("Москве", "LOC")], ("Маскве", "Москве", "substitution")),
    ("entity_typo", "pushkin_biography", "Биография Пушкина.", [("Пушкина", "PERSON")], ("Пушкна", "Пушкина", "deletion")),
    ("entity_typo", "msu_position", "Где расположен МГУ?", [("МГУ", "ORG")], ("МГГУ", "МГУ", "insertion")),
    ("entity_typo", "nizhny_foundation", "Когда основан Нижний Новгород?", [("Нижний Новгород", "LOC")], ("Новгроод", "Новгород", "transposition")),
    ("entity_typo", "lomonosov_works", "Работы Ломоносова.", [("Ломоносова", "PERSON")], ("Ломоносва", "Ломоносова", "deletion")),
    ("rare_correct", "chebyshev_research", "Исследования Пафнутия Чебышёва.", [("Пафнутия Чебышёва", "PERSON")]),
    ("rare_correct", "kovalevskaya_works", "Труды Софьи Ковалевской.", [("Софьи Ковалевской", "PERSON")]),
    ("rare_correct", "bove_buildings", "Постройки Осипа Бове.", [("Осипа Бове", "PERSON")]),
    ("rare_correct", "ustyuzhna_guide", "Путеводитель по Устюжне.", [("Устюжне", "LOC")]),
    ("rare_correct", "torzhok_museums", "Музеи Торжка.", [("Торжка", "LOC")]),
]
TEST = [
    ("clean", "korolev_achievements", "Достижения Сергея Павловича Королёва.", [("Сергея Павловича Королёва", "PERSON")]),
    ("clean", "spbu_foundation", "Когда основан Санкт-Петербургский государственный университет?", [("Санкт-Петербургский государственный университет", "ORG")]),
    ("clean", "kazan_universities", "Университеты Казани.", [("Казани", "LOC")]),
    ("clean", "rgo_petersburg", "Русское географическое общество в Санкт-Петербурге.", [("Русское географическое общество", "ORG"), ("Санкт-Петербурге", "LOC")]),
    ("clean", "discovery_types", "Какие бывают научные открытия?", []),
    ("ordinary_typo", "novosibirsk_universities", "История Новосибирска и его университетов.", [("Новосибирска", "LOC")], ("универсиетов", "университетов", "deletion")),
    ("ordinary_typo", "pushkin_books", "Книги об Александре Пушкине.", [("Александре Пушкине", "PERSON")], ("Книиги", "Книги", "insertion")),
    ("ordinary_typo", "hermitage_collections", "Коллекции Государственного Эрмитажа.", [("Государственного Эрмитажа", "ORG")], ("Коллекцеи", "Коллекции", "substitution")),
    ("ordinary_typo", "dubna_address", "Адрес института в Дубне.", [("Дубне", "LOC")], ("Адерс", "Адрес", "transposition")),
    ("ordinary_typo", "physics_experiments", "Эксперименты по физике.", [], ("Экспериметы", "Эксперименты", "deletion")),
    ("entity_typo", "gagarin_flight", "Первый полёт Юрия Гагарина.", [("Юрия Гагарина", "PERSON")], ("Гагарена", "Гагарина", "substitution")),
    ("entity_typo", "jinr_projects", "Проекты Объединённого института ядерных исследований.", [("Объединённого института ядерных исследований", "ORG")], ("институа", "института", "deletion")),
    ("entity_typo", "novosibirsk_map", "Карта Новосибирска.", [("Новосибирска", "LOC")], ("Новоссибирска", "Новосибирска", "insertion")),
    ("entity_typo", "mendeleev_law", "Периодический закон Менделеева.", [("Менделеева", "PERSON")], ("Менделееав", "Менделеева", "transposition")),
    ("entity_typo", "kfu_faculties", "Факультеты Казанского федерального университета.", [("Казанского федерального университета", "ORG")], ("Казаского", "Казанского", "deletion")),
    ("rare_correct", "chizhevsky_biography", "Биография Александра Чижевского.", [("Александра Чижевского", "PERSON")]),
    ("rare_correct", "buslaev_linguistics", "Лингвистические работы Фёдора Буслаева.", [("Фёдора Буслаева", "PERSON")]),
    ("rare_correct", "lobachevsky_legacy", "Наследие Николая Лобачевского.", [("Николая Лобачевского", "PERSON")]),
    ("rare_correct", "pereslavl_buildings", "Старинные здания Переславля-Залесского.", [("Переславля-Залесского", "LOC")]),
    ("rare_correct", "mezen_museum", "Краеведческий музей в городе Мезень.", [("Мезень", "LOC")]),
]


def serialize(specs, split):
    records = []
    for i, spec in enumerate(specs, 1):
        group, family, correct, mentions, *error = spec
        original = correct
        typos = []
        if error:
            wrong, right, operation = error[0]
            assert correct.count(right) == 1
            start = correct.index(right)
            original = correct[:start] + wrong + correct[start + len(right):]
            typos = [dict(start=start, end=start + len(wrong), wrong=wrong, correct=right, operation=operation)]
        correct_entities, original_entities = [], []
        for mention, kind in mentions:
            assert correct.count(mention) == 1
            start = correct.index(mention)
            correct_entities.append(dict(start=start, end=start + len(mention), text=mention, type=kind))
            original_mention = mention
            if error:
                wrong, right, _ = error[0]
                typo_start = correct.index(right)
                if start <= typo_start and typo_start + len(right) <= start + len(mention):
                    local = typo_start - start
                    original_mention = mention[:local] + wrong + mention[local + len(right):]
                elif typo_start < start:
                    start += len(wrong) - len(right)
            assert original[start:start + len(original_mention)] == original_mention
            original_entities.append(dict(start=start, end=start + len(original_mention), text=original_mention, type=kind))
        record = dict(id=f"{split}_{i:02d}", split=split, group=group, family_id=family,
                      original_text=original, correct_text=correct, typos=typos,
                      original_entities=original_entities, correct_entities=correct_entities)
        if group == "rare_correct":
            inside = {"chebyshev_research", "kovalevskaya_works", "lobachevsky_legacy", "pereslavl_buildings"}
            record["rare_case"] = "selected_corpus_topic" if family in inside else "outside_selected_topics"
        records.append(record)
    return records


def main():
    path = ROOT / "data/queries"
    path.mkdir(parents=True, exist_ok=True)
    for split, specs in (("dev", DEV), ("test", TEST)):
        records = serialize(specs, split)
        (path / f"queries_{split}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print("Saved 20 dev + 20 test manually authored queries")


if __name__ == "__main__":
    main()
