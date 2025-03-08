import torch
import re
import pandas as pd
import streamlit as st
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


# Интерфейс Streamlit
st.title("Классификация высказываний на русском языке")
st.markdown(
    f"""Пример данных (можно заменить на реальные высказывания)
            
    - Непорядочное отношение к своим работникам.
            
    - Я недоволен сервисом, ожидал лучшего.
            
    - Сегодня погода отличная, настроение супер!
        
    - asdfasfdafdasf"""
)

user_input = st.text_area("Введите текст для классификации:")
if st.button("Классифицировать"):
    if user_input.strip():
        processed_text = preprocess_text(user_input)
        prediction = classify_texts([processed_text])
        st.write(f"**Категория:** {prediction}")
    else:
        st.warning("Введите текст перед классификацией.")
