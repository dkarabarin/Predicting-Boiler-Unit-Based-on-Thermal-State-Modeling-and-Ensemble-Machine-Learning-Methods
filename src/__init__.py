# src/__init__.py
from .config import SENSORS, FORECAST_HOURS, STEPS_FORWARD, MODELS_DIR
from .features import FeatureCreator
from .predictor import ModelPredictor