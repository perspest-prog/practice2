import json
from pathlib import Path
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'mnist.npz'
MODEL = ROOT / 'output' / 'model.npz'
URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'


class Neuron:
    """Один входной элемент. Вес хранится в общем массиве сети."""
    def __init__(self, weights, index):
        self.weights, self.index = weights, index

    @property
    def weight(self):
        return float(self.weights[self.index])

    def calculate(self, value):
        return self.weight * value

    def update(self, delta):
        self.weights[self.index] += delta


class Network:
    def __init__(self, seed=42):
        self.weights = np.random.default_rng(seed).uniform(-0.01, 0.01, 784)
        self.neurons = [Neuron(self.weights, i) for i in range(784)]
        self.studyCoeff = 0.01
        self.correctCoeff = 0.5

    def calculate(self, pixels):
        # Векторная запись суммы neuron.calculate(pixel) для всех пикселей.
        return np.asarray(pixels) @ self.weights

    def predict(self, pixels):
        return (self.calculate(pixels) >= self.correctCoeff).astype(np.int8)

    def update(self, pixels, expected):
        error = int(expected) - int(self.predict(pixels))
        if error:
            # Эквивалент neuron.update(studyCoeff * error * pixel).
            self.weights += self.studyCoeff * error * pixels
        return bool(error)

    def save(self, path=MODEL):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=self.weights, studyCoeff=self.studyCoeff,
                 correctCoeff=self.correctCoeff, digit=9, pixel_threshold=127)

    @classmethod
    def load(cls, path=MODEL):
        with np.load(path, allow_pickle=False) as data:
            net = cls()
            net.studyCoeff = float(data['studyCoeff'])
            net.correctCoeff = float(data['correctCoeff'])
            net.weights[:] = data['weights']
        return net


def load_data():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        print('Скачивание MNIST...', flush=True)
        temporary = DATA.with_suffix('.download')
        try:
            urllib.request.urlretrieve(URL, temporary)
            temporary.replace(DATA)
        finally:
            temporary.unlink(missing_ok=True)
    with np.load(DATA, allow_pickle=False) as data:
        return tuple(data[key] for key in ('x_train', 'y_train', 'x_test', 'y_test'))


def pixels(images):
    return (images.reshape(-1, 784) > 127).astype(np.float64)


def metrics(expected, predicted):
    positive = expected == 1
    tp = int(np.sum(positive & (predicted == 1)))
    fn = int(np.sum(positive & (predicted == 0)))
    fp = int(np.sum(~positive & (predicted == 1)))
    tn = int(np.sum(~positive & (predicted == 0)))
    recall, specificity = tp / (tp + fn), tn / (tn + fp)
    return dict(accuracy=(tp + tn) / len(expected), recall=recall,
                specificity=specificity, precision=tp / max(tp + fp, 1),
                balanced_accuracy=(recall + specificity) / 2,
                tp=tp, fn=fn, fp=fp, tn=tn)


def report(title, result):
    print(f"{title}: точность {result['accuracy']:.2%}; "
          f"найдено девяток {result['recall']:.2%}; "
          f"верно отклонено других цифр {result['specificity']:.2%}")


def split(labels, rng):
    train, validation = [], []
    # Стратификация по всем десяти цифрам, около 10% на валидацию.
    for digit in range(10):
        indices = rng.permutation(np.flatnonzero(labels == digit))
        count = len(indices) // 10
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return np.array(train), np.array(validation)


def train(seed=42, epochs=100):
    images, labels, test_images, test_labels = load_data()
    rng = np.random.default_rng(seed)
    training, validation = split(labels, rng)
    x, y = pixels(images[training]), (labels[training] == 9).astype(np.int8)
    vx, vy = pixels(images[validation]), (labels[validation] == 9).astype(np.int8)
    net = Network(seed)
    best_score, best_weights, history = -1, None, []
    print(f'Обучение: {len(y)}; валидация: {len(vy)}; тест: {len(test_labels)}', flush=True)
    for epoch in range(1, epochs + 1):
        # Все обучающие изображения в случайном порядке. Усреднение весов
        # в течение эпохи уменьшает влияние последних примеров.
        order = rng.permutation(len(y))
        total_weights = np.zeros(784)
        errors = 0
        for i in order:
            errors += net.update(x[i], y[i])
            total_weights += net.weights
        net.weights[:] = total_weights / len(order)
        result = metrics(vy, net.predict(vx))
        history.append(dict(epoch=epoch, mistakes=errors, **result))
        report(f'Эпоха {epoch}, ошибок обучения {errors}', result)
        if result['accuracy'] > best_score:
            best_score, best_weights = result['accuracy'], net.weights.copy()
        if result['accuracy'] >= 0.95:
            print('Цель 95% на валидации достигнута.')
            break
    else:
        print('Лимит эпох достигнут. Цель 95% не достигнута, сохранена лучшая модель.')
    net.weights[:] = best_weights
    net.save()
    test = metrics((test_labels == 9).astype(np.int8), net.predict(pixels(test_images)))
    report('Независимый тест', test)
    summary = dict(digit=9, seed=seed, studyCoeff=net.studyCoeff, correctCoeff=net.correctCoeff,
                   train_size=len(y), validation_size=len(vy), test_size=len(test_labels),
                   best_validation_accuracy=best_score, test=test, history=history)
    (MODEL.parent / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Модель: {MODEL}')


def main():
    if not MODEL.exists():
        print('Модель не найдена. Запускаю обучение...')
        train()
    net = Network.load()
    _, _, images, labels = load_data()
    index = int(input('Введите индекс изображения (примеры девяток - 7, 9, 12, 99, 113, 320, 359): '))

    if not 0 <= index < len(labels):
        raise ValueError(f'Индекс должен быть от 0 до {len(labels) - 1}')
    image = images[index]
    for row in image:
        print(''.join('██' if p > 127 else '  ' for p in row))
    score = float(net.calculate(pixels(image)[0]))

    print(f'Индекс: {index}. Настоящая цифра: {labels[index]}')
    print(f'Сумма: {score:.3f}. Порог: {net.correctCoeff}. Ответ: ' + ('9' if score >= net.correctCoeff else 'не 9'))


if __name__ == '__main__':
    main()
