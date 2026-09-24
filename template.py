import json
import random

import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


# ==========================================
# Загрузка данных
# ==========================================

with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)


# ==========================================
# Подготовка обучающих данных
# ==========================================

patterns = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        labels.append(intent["tag"])


# ==========================================
# Создание нейронной сети
# ==========================================

model = Pipeline([
    (
        "vectorizer",
        TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b"
        )
    ),
    (
        "classifier",
        MLPClassifier(
            hidden_layer_sizes=(16, 16),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42
        )
    )
])


# ==========================================
# Обучение модели
# ==========================================

print("Обучение модели...")

model.fit(patterns, labels)

print("Модель успешно обучена!")


# ==========================================
# Сохранение модели
# ==========================================

joblib.dump(model, "model.pkl")

print("Модель сохранена в model.pkl")


# ==========================================
# Определение намерения пользователя
# ==========================================

def predict_intent(text):
    probabilities = model.predict_proba([text])[0]

    best_index = np.argmax(probabilities)

    tag = model.classes_[best_index]
    confidence = probabilities[best_index]

    return tag, confidence


# ==========================================
# Чат с пользователем
# ==========================================

def chat():
    print()
    print("Бот готов к работе!")
    print("Введите 'quit', чтобы выйти.")
    print()

    while True:
        inp = input("Вы: ")

        if inp.lower() == "quit":
            print("Бот: До свидания!")
            break

        tag, confidence = predict_intent(inp)

        # Если модель недостаточно уверена
        if confidence < 0.3:
            print("Бот: Извините, я не понял ваш вопрос.")
            continue

        # Ищем нужный intent
        responses = None

        for intent in data["intents"]:
            if intent["tag"] == tag:
                responses = intent["responses"]
                break

        # Выбираем случайный ответ
        if responses:
            print("Бот:", random.choice(responses))
        else:
            print("Бот: Я пока не знаю, что ответить.")


# ==========================================
# Запуск программы
# ==========================================

chat()