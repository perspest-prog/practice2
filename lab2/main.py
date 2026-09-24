from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pymorphy3

QUESTIONS = [
    ("осмотр автомобиля", "Осмотр автомобиля проводят в мастерской на первом этаже."),
    ("ремонт мобильного телефона", "Обратитесь в сервис телефонов, кабинет 102."),
    ("ремонт персонального компьютера", "Ремонт ПК выполняют в кабинете 103."),
    ("настройка электронной почты", "Откройте настройки почты и укажите адрес сервера mail.example.org."),
    ("восстановление пароля", "Нажмите «Забыли пароль» на странице входа."),
    ("адрес библиотеки", "Библиотека находится в корпусе 2, кабинет 201."),
    ("номер телефона", "Телефон учебного справочного центра: 000-00-00."),
    ("расписание занятий", "Расписание размещено на стенде рядом с деканатом."),
]


ROOT = Path(__file__).resolve().parent


def tokenize(text):
    """Сохраняем слова, числа и дефис внутри слова; ё приравниваем к е."""
    return re.findall(r"[а-яa-z0-9]+(?:-[а-яa-z0-9]+)*", text.lower().replace("ё", "е"))


@dataclass
class Analysis:
    words: list
    lemmas: list
    pairs: list
    steps: list
    units: tuple


class ThesaurusSearch:
    def __init__(self, questions=QUESTIONS, directory=ROOT / "thesaurus"):
        self.morph = pymorphy3.MorphAnalyzer()
        self.stop_words = {
            word
            for item in ET.parse(directory / "service_words.xml").getroot()
            for word in tokenize(item.text or "")
        }
        # Одному слову могут соответствовать несколько значений. Храним все.
        self.index = {}
        self.labels = {}
        self.entry_count = 0
        # В потоке читаем большой XML, не держим всё XML-дерево в памяти.
        for _, item in ET.iterparse(directory / "senses.xml", events=("end",)):
            if item.tag != "Item":
                continue
            name = tuple(tokenize(item.get("name", "")))
            lemma = tuple(tokenize(item.get("lemma", "")))
            # Общее понятие связывает разные части речи, например
            # восстановление (141399-N) и восстановить (141399-V).
            concept = item.get("concept_id")
            group = item.get("synset_id")
            if concept:
                key = "concept:" + concept
            elif group:
                key = "synset:" + group
            else:
                key = "lemma:" + " ".join(lemma or name)
            self.labels.setdefault(key, " ".join(lemma or name))
            for phrase in (name, lemma):
                if phrase:
                    self.index.setdefault(phrase, set()).add(key)
            self.entry_count += 1
            item.clear()
        self.questions = [(question, answer, self.analyze(question)) for question, answer in questions]

    @lru_cache(maxsize=10000)
    def lemmatize(self, word):
        # Берём наиболее вероятный морфологический разбор.
        return self.morph.parse(word)[0].normal_form.replace("ё", "е")

    def lookup(self, words, lemmas):
        # Сначала точная словарная форма: это важно для сокращений, например ПК.
        return self.index.get(tuple(words)) or self.index.get(tuple(lemmas))

    def analyze(self, text):
        words = [word for word in tokenize(text) if word not in self.stop_words]
        lemmas = [self.lemmatize(word) for word in words]
        pairs = [" ".join(words[i:i + 2]) for i in range(len(words) - 1)]
        steps, units, covered = [], [], set()
        # Проверяем все перекрывающиеся двусловия.
        for i, pair in enumerate(pairs):
            found = self.lookup(words[i:i + 2], lemmas[i:i + 2])
            if found:
                units.append(frozenset(found))
                covered.update((i, i + 1))
                steps.append((pair, frozenset(found)))
            else:
                steps.append((pair, None))
        # Слова известного выражения уже представлены его смыслом.
        # Для остальных выполняем предусмотренный заданием поиск по одному слову.
        for i, word in enumerate(words):
            if i not in covered:
                found = self.lookup([word], [lemmas[i]])
                unit = frozenset(found or {"word:" + lemmas[i]})
                units.append(unit)
                steps.append((word, unit))
        return Analysis(words, lemmas, pairs, steps, tuple(dict.fromkeys(units)))

    @staticmethod
    def equivalent(left, right):
        """Полное взаимно-однозначное сопоставление смысловых единиц.

        У единицы могут быть альтернативные значения. Достаточно общего понятия,
        но один элемент вопроса нельзя использовать для двух элементов запроса.
        Частичное совпадение и совпадение только одного слова не дают ответ.
        """
        if not left or len(left) != len(right):
            return False
        matched = {}

        def assign(i, visited):
            for j, target in enumerate(right):
                if j not in visited and left[i] & target:
                    visited.add(j)
                    if j not in matched or assign(matched[j], visited):
                        matched[j] = i
                        return True
            return False

        return all(assign(i, set()) for i in range(len(left)))

    def search(self, text):
        analysis = self.analyze(text)
        matches = [(question, answer) for question, answer, stored in self.questions
                   if self.equivalent(analysis.units, stored.units)]
        return analysis, matches

    def describe(self, unit):
        return " / ".join(f"{self.labels.get(key, key.removeprefix('word:'))} [{key}]"
                          for key in sorted(unit))

    def reply(self, text, verbose=False):
        analysis, matches = self.search(text)
        if verbose:
            print("После очистки:", " ".join(analysis.words) or "(пусто)")
            print("Начальные формы:", ", ".join(analysis.lemmas) or "(нет)")
            print("Двусловия:", "; ".join(analysis.pairs) or "(нет)")
            for phrase, unit in analysis.steps:
                print(f"  {phrase} → {self.describe(unit) if unit else 'пара не найдена'}")
            print("Смысловые единицы без повторов:")
            for unit in analysis.units:
                print("  ", self.describe(unit))
        if not analysis.units:
            print("Ответ: Введите запрос, содержащий значимые слова.")
        elif not matches:
            print("Ответ: Подходящий ответ не найден. Попробуйте переформулировать запрос.")
        elif len(matches) > 1:
            print("Ответ: Уточните запрос. Подходят вопросы:")
            for question, _ in matches:
                print("  -", question)
        else:
            if verbose:
                print("Совпавший вопрос:", matches[0][0])
            print("Ответ:", matches[0][1])


def main():
    print("Загрузка тезауруса...")
    try:
        engine = ThesaurusSearch()
    except (OSError, ET.ParseError) as error:
        print(f"Не удалось прочитать XML тезауруса: {error}\n")
        return

    print(f"Загружено записей: {engine.entry_count}. Вопросов в базе: {len(engine.questions)}.")
    print("Введите вопрос. /list — вопросы базы, /exit — выход.")

    while True:
        try:
            query = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break
        if query.lower() in {"/exit", "exit", "quit", "выход"}:
            break
        if query == "/list":
            for question, _, _ in engine.questions:
                print(" -", question)
            continue
        engine.reply(query, verbose=False)


if __name__ == "__main__":
    main()
