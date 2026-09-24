import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt_tab')
nltk.download('punkt') # Раскомментировать при первом запуске
nltk.download('stopwords') # Раскомментировать при первом запуске

def preprocess_text(text):
    # Токенизация
    tokens = word_tokenize(text)
    # Приведение к нижнему регистру и удаление пунктуации/стоп-слов
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('russian')]
    return tokens

from pypdf import PdfReader
reader = PdfReader("./voina-i-mir.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

processed_text = preprocess_text(text)

from gensim.models import Word2Vec

# Обучение модели Skip-gram
model_sg = Word2Vec(
    sentences=[processed_text],  # Текст для обучения
    vector_size=100,             # Размерность вектора слова
    window=5,                    # Размер контекстного окна
    min_count=5,                 # Минимальная частота появления слова
    sg=1,                        # 1 = Skip-gram
    workers=4                    # Количество потоков
)
# Обучение модели CBOW
model_cbow = Word2Vec(
    sentences=[processed_text],
    vector_size=100, window=5,
    min_count=5,
    sg=0,
    workers=4
)

import gensim.downloader as api

pretrained_model = api.load('word2vec-ruscorpora-300')

def print_similar_words(model, model_name, words):
    print(f"\n===== {model_name} =====")

    for word in words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)

            print(f"\nСлово: {word}")
            for similar_word, similarity in similar:
                print(f"  {similar_word}: {similarity:.4f}")
        else:
            print(f"\nСлово '{word}' отсутствует в модели")

def print_similar_words_pretrained(model, model_name, words):
    print(f"\n===== {model_name} =====")

    for word in words:
        if f"{word}_NOUN" in model:
            similar = model.most_similar(f"{word}_NOUN", topn=5)

            print(f"\nСлово: {word}")
            for similar_word, similarity in similar:
                print(f"  {similar_word}: {similarity:.4f}")
        else:
            print(f"\nСлово '{word}' отсутствует в модели")

keywords = ['война', 'мир', 'человек', 'земля', 'князь']
print_similar_words(model_sg, "Skip-gram — Война и мир", keywords)
print_similar_words(model_cbow, "CBOW — Война и мир", keywords )
print_similar_words_pretrained(pretrained_model, "Word2Vec RusCorpora", keywords)

result_sg = model_sg.wv.most_similar(
    positive=['мужчина', 'княгиня'],
    negative=['женщина'],
    topn=5
)

result_cbow = model_cbow.wv.most_similar(
    positive=['мужчина', 'княгиня'],
    negative=['женщина'],
    topn=5
)

result_pretrained = pretrained_model.most_similar(
    positive=['мужчина_NOUN', 'княгиня_NOUN'],
    negative=['женщина_NOUN'],
    topn=5
)

print("Skip-gram:", result_sg)
print("CBOW:", result_cbow)
print("Pretrained:", result_pretrained)

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(vectors_model, groups, title, output_path, suffix=''):
    words = [word for group in groups.values() for word in group]
    vectors = np.asarray([vectors_model[word + suffix] for word in words])
    if len(words) < 3:
        raise ValueError('Для t-SNE необходимо не менее трех слов.')
    points = TSNE(
        n_components=2,
        perplexity=min(5, len(words) - 1),
        metric='cosine',
        init='random',
        learning_rate=200.0,
        random_state=42,
    ).fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(12, 9))
    start = 0
    for (theme, group), color, marker in zip(
        groups.items(), ['#2166ac', '#d95f02'], ['o', '^']
    ):
        end = start + len(group)
        ax.scatter(points[start:end, 0], points[start:end, 1],
                   color=color, marker=marker, s=65, label=theme, alpha=0.85)
        start = end
    for word, (x, y) in zip(words, points):
        ax.annotate(word, (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=10)
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.margins(0.18)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig


tsne_groups = {
    'Люди и социальные роли': [
        'князь', 'граф', 'графиня', 'княгиня', 'император',
        'царь', 'офицер', 'солдат', 'генерал', 'княжна',
    ],
    'Военные понятия': [
        'война', 'армия', 'сражение', 'оружие', 'войско',
        'полк', 'атака', 'победа', 'бой', 'ружье',
    ],
}

plots_dir = Path(__file__).resolve().parent / 'tsne_plots'
plots_dir.mkdir(exist_ok=True)
for vectors_model, title, filename, suffix in [
    (model_sg.wv, 'Skip-gram — Война и мир', 'skip_gram.png', ''),
    (model_cbow.wv, 'CBOW — Война и мир', 'cbow.png', ''),
    (pretrained_model, 'Word2Vec RusCorpora', 'ruscorpora.png', '_NOUN'),
]:
    plot_tsne(vectors_model, tsne_groups, title, plots_dir / filename, suffix)

plt.show()
