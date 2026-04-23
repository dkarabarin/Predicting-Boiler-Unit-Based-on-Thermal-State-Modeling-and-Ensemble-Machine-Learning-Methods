# src/config.py
import os

SENSORS = {
    '10HAH01CT103': 'средние ширмы',
    '10HAH01CT102': 'средние ширмы',
    '10HAH12CT101': 'ширмы',
    '10HAH12CT104': 'ширмы',
    '10HAH12CT110': 'пароперегреватель',
    '10HAH12CT108': 'пароперегреватель',
    '10HAH12CT106': 'пароперегреватель',
    '10HAH11CT114': 'пароперегреватель',
    '10HAH11CT113': 'пароперегреватель',
    '10HAH12CT116': 'ширмы',
    '10HAH12CT117': 'ширмы'
}

PHYSICS = {
    'средние ширмы': 1.0,
    'ширмы': 1.2,
    'пароперегреватель': 1.5
}

# Константы стали 12Х1МФ
A_STEEL = 1.2e-12
U0_STEEL = 480000
GAMMA_STEEL = 0.35
R_GAS = 8.314

FORECAST_HOURS = 24
STEPS_FORWARD = FORECAST_HOURS * 6

MODELS_DIR = "trained_models_best"
DATA_PATH = "data/temperature_data.csv"

os.makedirs(MODELS_DIR, exist_ok=True)