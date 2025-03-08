import torch
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def preprocess_text(text):
    """Очищает текст от лишних символов и приводит к единому формату."""
    text = text.lower().strip()
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9.,!?; ]+", "", text)
    return text


# Загрузка дообученной модели
MODEL_PATH = "./fine_tuned_llm-08-03-2025"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Категории, которые использовались при обучении (пример)
categories = ["Нейтрально", "Позитивный", "Негативный"]
# 0: neutral
# 1: positive
# 2: negative


def classify_texts(texts):
    """Классифицирует список текстов."""
    inputs = tokenizer(
        texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = probabilities > 0.4  # Бинаризация предсказаний

    results = []
    for pred in predictions:
        labels = [categories[i] for i, val in enumerate(pred) if val]
        if not labels:
            labels = ["Нейтрально"]  # Если ни одна категория не выбрана
        results.append("; ".join(labels))

    return results


# Пример данных (можно заменить на реальные высказывания)
data = pd.Series(
    [
        "Непорядочное отношение к своим работникам.",
        "Я недоволен сервисом, ожидал лучшего.",
        "Сегодня погода отличная, настроение супер!",
        "asdfasfdafdasf",
    ]
)

# Предобработка данных
processed_data = data.apply(preprocess_text)

# Классификация
predictions = classify_texts(processed_data.tolist())

# Вывод результатов
for text, category in zip(data, predictions):
    print(f"{text} — {category}")
