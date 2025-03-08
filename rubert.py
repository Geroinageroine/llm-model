import pandas as pd
import torch
import re
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset


def preprocess_text(text):
    """Очищает текст от лишних символов и приводит к единому формату."""
    text = text.lower().strip()
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9.,!?; ]+", "", text)
    return text


# Загружаем альтернативный датасет для анализа тональности
DATASET_NAME = "MonoHime/ru_sentiment_dataset"
dataset = load_dataset(DATASET_NAME)

# Подготавливаем категории
categories = list(set(dataset["train"]["sentiment"]))
category_to_id = {cat: i for i, cat in enumerate(categories)}


def encode_labels(example):
    example["labels"] = category_to_id[example["sentiment"]]
    return example


dataset = dataset.map(encode_labels)


def tokenize_function(example):
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    return tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=128
    )


dataset = dataset.map(tokenize_function, batched=True)

# Загружаем модель
model = AutoModelForSequenceClassification.from_pretrained(
    "cointegrated/rubert-tiny2", num_labels=len(categories)
)

# Определяем параметры обучения
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# Запуск обучения
trainer.train()

# Сохраняем обученную модель
model.save_pretrained("./fine_tuned_rubert_sentiment")
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
tokenizer.save_pretrained("./fine_tuned_rubert_sentiment")
print("Модель успешно дообучена и сохранена!")
