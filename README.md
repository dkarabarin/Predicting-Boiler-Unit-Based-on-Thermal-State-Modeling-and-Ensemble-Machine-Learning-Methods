# 🔥 Прогнозирование остановов оборудования ТЭС

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)](https://pytorch.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)](https://fastapi.tiangolo.com/) [![CUDA](https://img.shields.io/badge/CUDA-12.6-brightgreen)](https://developer.nvidia.com/cuda-toolkit) [![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Система прогнозирования остановов котлоагрегата БКЗ-500-140 на основе температурных данных с использованием ML и нейросетей. Прогноз на 24 часа и до года вперед.

---

## 📋 Содержание
- [Обзор](#обзор)
- [Структура проекта](#структура-проекта)
- [Быстрый старт](#быстрый-старт)
- [Установка](#установка)
- [Обучение моделей](#обучение-моделей)
- [Запуск сервера](#запуск-сервера)
- [API](#api)
- [Веб-интерфейс](#веб-интерфейс)
- [Модели и признаки](#модели-и-признаки)
- [Результаты](#результаты)
- [Технологии](#технологии)

---

## Обзор

**Задача:** Прогнозирование остановок котла БКЗ-500-140 по данным 11 датчиков температуры поверхностей нагрева.

**Решение:** 
- 🚀 Обучение на GPU (RTX 3060, CUDA 12.6)
- 🤖 6 ML моделей с GridSearch подбором гиперпараметров
- 🧠 3 нейросетевые архитектуры (DNN, Wide&Deep, Attention)
- 📊 Автовыбор лучшей модели (ML или NN) для каждого датчика
- 🌐 Веб-интерфейс на FastAPI с интерактивными графиками Plotly
- 📁 Загрузка CSV через Drag & Drop
- 📅 Прогноз на год вперед с помесячной сводкой

**Данные:** 5,738,001 записей, 2013-2018 гг, 10-минутные интервалы, 11 датчиков температуры.

**Результат:** Средний AUC на тесте 2018 года = **0.9546**, средний F1 = **0.9055**.

---

## Структура проекта
tes-prediction/
├── main.py # FastAPI сервер (запуск: python main.py)
├── full.ipynb # Jupyter ноутбук обучения ML+NN
├── fix_nn_models.py # Исправление нейросетей для CPU
├── requirements.txt # Зависимости Python
├── README.md # Документация
├── temperature_data.csv # Исходные данные (5.7M записей)
├── uploads/ # Загрузки пользователей
├── logs/ # Логи сервера
└── trained_models_best/ # Обученные модели (22 файла)
├── 10HAH01CT103.pkl # ML модель (GradientBoosting, AUC=0.8896)
├── 10HAH01CT103n.pkl # Нейросеть (WideDeepNet, AUC=0.7919)
├── 10HAH01CT102.pkl # ML модель (RandomForest, AUC=0.9530)
├── 10HAH01CT102n.pkl # Нейросеть (WideDeepNet, AUC=0.9399)
├── ... # 11 ML + 11 NN = 22 модели
├── training_results.json # Метрики всех моделей в JSON
└── training_summary.csv # Сводная таблица результатов

code
Copy

---

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone https://github.com/your-username/tes-prediction.git
cd tes-prediction

# 2. Создать и активировать виртуальное окружение
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Установить зависимости
pip install -r requirements.txt
