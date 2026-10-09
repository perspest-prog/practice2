"""CLI for correction, both NER approaches and pipelines A/B."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from spelling import SpellCorrector, levenshtein, levenshtein_matrix
from text_utils import normalize
from ner import ModelLoadError, create_recognizer
from pipeline import process

ROOT = Path(__file__).resolve().parent


def format_text(output):
    """Present CLI results without changing processing or JSON records."""
    if "matrix" in output:
        lines = [f'Слова: «{output["left"]}» → «{output["right"]}»',
                 f'Расстояние Левенштейна: {output["distance"]}',
                 'Операции: вставка, удаление, замена; стоимость 1.',
                 'Матрица (строки — первое слово, столбцы — второе):']
        width = max(2, len(str(max(len(output["left"]), len(output["right"])))))
        lines.append('    ' + ' '.join(f'{c:>{width}}' for c in ['∅', *output['right']]))
        for label, row in zip(['∅', *output['left']], output['matrix']):
            lines.append(f'{label:>3} ' + ' '.join(f'{n:>{width}}' for n in row))
        return '\n'.join(lines) + '\n'

    config = output['config']
    lines = []
    if 'recognizer' in config:
        name = config['recognizer']['name']
        names = {'dictionary_and_rules': 'справочник и правила',
                 'slovnet_ner_news_v1': 'модель Slovnet'}
        lines.append('Распознаватель: ' + names.get(name, name))
    if 'order' in config:
        protection = 'включена' if config['protect_entities'] else 'отключена'
        lines.append(f'Порядок: {config["order"]}; защита сущностей: {protection}')
    if 'threshold' in config:
        frequencies = 'включены' if config['use_frequencies'] else 'отключены'
        lines.append(f'Порог: {config["threshold"]}; частоты: {frequencies}')
    if not output['results']:
        lines.append('Нет запросов для обработки.')
    for result in output['results']:
        lines.extend(['', f'Запрос {result["id"]}', f'Исходный:    {result["original_text"]}'])
        if 'proposed_text' in result:
            lines.append(f'Предложенный: {result["proposed_text"]}')
        if result.get('message'):
            lines.append(result['message'])
        if 'changes' in result:
            lines.append('Замены (границы в исходном тексте):')
            if not result['changes']:
                lines.append('  Нет замен.')
            for change in result['changes']:
                lines.append(f'  [{change["start"]}, {change["end"]}) '
                             f'«{change["old"]}» → «{change["new"]}»; расстояние {change["distance"]}')
                alternatives = change['alternatives']
                if alternatives:
                    lines.append('    Альтернативы: ' + '; '.join(
                        f'«{c["text"]}» (расстояние {c["distance"]}, частота {c["frequency"]})'
                        for c in alternatives))
            unknown = [t['original'] for t in result['tokens'] if t['status'] == 'no_candidates']
            if unknown:
                lines.append('Без подходящих кандидатов: ' + ', '.join(f'«{word}»' for word in unknown))
        if result.get('config', {}).get('protect_entities'):
            lines.append('Защищённые сущности (исходный текст):')
            for entity in result['initial_entities']:
                lines.append(f'  «{entity["text"]}» — {entity["type"]}, '
                             f'[{entity["start"]}, {entity["end"]})')
            if not result['initial_entities']:
                lines.append('  Нет.')
        if 'entities' in result:
            positions = 'предложенный текст' if result['entity_positions'] == 'proposed_text' else 'исходный текст'
            lines.append(f'Сущности ({positions}):')
            for entity in result['entities']:
                lines.append(f'  «{entity["text"]}» — {entity["type"]}, '
                             f'[{entity["start"]}, {entity["end"]})')
            if not result['entities']:
                lines.append('  Не найдены.')
    return '\n'.join(lines) + '\n'


def read_queries(path):
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("Файл JSON должен содержать массив строк или записей original_text")
        rows = []
        for i, item in enumerate(data, 1):
            if isinstance(item, str):
                rows.append((str(i), item))
            elif isinstance(item, dict) and isinstance(item.get("original_text"), str):
                rows.append((item.get("id", str(i)), item["original_text"]))
            else:
                raise ValueError(f"Некорректная запись запроса № {i}")
        return rows
    return [(str(i), line) for i, line in enumerate(path.read_text().splitlines(), 1)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Лабораторная 4: опечатки и именованные сущности")
    sub = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (("correct", "Исправить отдельный запрос или набор"),
                               ("ner", "Найти сущности без коррекции"),
                               ("process", "Совместная обработка в порядке A/B")):
        operation = sub.add_parser(command, help=help_text)
        source = operation.add_mutually_exclusive_group(required=True)
        source.add_argument("--query")
        source.add_argument("--file", type=Path)
        operation.add_argument("--dictionary-dir", type=Path, default=ROOT / "data/dictionaries")
        operation.add_argument("--output", type=Path)
        operation.add_argument("--format", choices=["json", "text"], default="json",
                               help="Формат вывода: JSON или читаемый текст")
        if command != "ner":
            operation.add_argument("--threshold", type=int, choices=[1, 2], default=1)
            operation.add_argument("--no-frequencies", action="store_true")
        if command != "correct":
            operation.add_argument("--recognizer", choices=["dictionary", "model"], default="dictionary")
            operation.add_argument("--model-dir", type=Path, default=ROOT / "models")
        if command == "process":
            operation.add_argument("--order", choices=["A", "B"], default="A")
            operation.add_argument("--no-protection", action="store_true", help="Отключить защиту в B; повторный NER сохранится")
    distance = sub.add_parser("distance", help="Показать расстояние и таблицу динамического программирования")
    distance.add_argument("left")
    distance.add_argument("right")
    distance.add_argument("--normalize", action="store_true", help="Игнорировать регистр и различие ё/е")
    distance.add_argument("--format", choices=["json", "text"], default="json",
                          help="Формат вывода: JSON или читаемая матрица")
    args = parser.parse_args(argv)
    try:
        if args.command == "distance":
            left, right = (normalize(args.left), normalize(args.right)) if args.normalize else (args.left, args.right)
            output = dict(left=left, right=right, normalized=args.normalize,
                          distance=levenshtein(left, right), matrix=levenshtein_matrix(left, right),
                          operations="insertion, deletion, substitution; cost=1; no transposition")
        else:
            if args.file and args.output and args.file.resolve() == args.output.resolve():
                raise ValueError("Выходной файл не должен перезаписывать исходные запросы")
            queries = read_queries(args.file) if args.file else [("query", args.query)]
            paths = [ROOT / "text_utils.py", ROOT / "main.py"]
            config = {}
            if args.command != "ner":
                corrector = SpellCorrector.from_files(args.dictionary_dir, threshold=args.threshold,
                                                      use_frequencies=not args.no_frequencies)
                paths.extend([args.dictionary_dir / "word_dictionary.json", args.dictionary_dir / "word_details.json", ROOT / "spelling.py"])
                config.update(threshold=args.threshold, use_frequencies=not args.no_frequencies)
            if args.command != "correct":
                recognizer = create_recognizer(args.recognizer, args.dictionary_dir, args.model_dir)
                paths.append(ROOT / "ner.py")
                if args.recognizer == "dictionary":
                    paths.append(args.dictionary_dir / "entity_dictionary.json")
                config["recognizer"] = recognizer.metadata()
            if args.command == "correct":
                results = [dict(id=qid, **corrector.correct(text)) for qid, text in queries]
                stage = "spelling_only"
            elif args.command == "ner":
                results = [dict(id=qid, original_text=text, entities=recognizer.recognize(text),
                                entity_positions="original_text", message="Пустой ввод" if not text.strip() else "")
                           for qid, text in queries]
                stage = "ner_only"
            else:
                config.update(order=args.order, protect_entities=args.order == "B" and not args.no_protection)
                paths.append(ROOT / "pipeline.py")
                results = [dict(id=qid, **process(text, corrector, recognizer, args.order, not args.no_protection))
                           for qid, text in queries]
                stage = "joint_pipeline"
            output = dict(stage=stage, input_file=str(args.file) if args.file else None,
                          dictionary_dir=str(args.dictionary_dir), config=config,
                          source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                          results=results)
        serialized = (format_text(output) if args.format == "text"
                      else json.dumps(output, ensure_ascii=False, indent=2) + "\n")
        if args.command != "distance" and args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized)
            print(f"Результаты сохранены: {args.output}")
        else:
            print(serialized, end="")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ModelLoadError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
