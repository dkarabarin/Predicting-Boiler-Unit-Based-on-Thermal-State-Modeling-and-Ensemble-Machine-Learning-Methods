# src/predictor.py
import os
import joblib
import numpy as np
import pandas as pd
import torch
from .config import MODELS_DIR, SENSORS
from .models_nn import NEURAL_MODELS
from .features import FeatureCreator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelPredictor:
    """Загрузка моделей и прогнозирование"""
    
    def __init__(self):
        self.models_cache = {}
        self.feature_creator = FeatureCreator()
        self._load_models()
    
    def _load_models(self):
        """Загрузка всех моделей из папки"""
        if not os.path.exists(MODELS_DIR):
            print(f"⚠️ Папка {MODELS_DIR} не найдена")
            return
        
        for filename in os.listdir(MODELS_DIR):
            if not filename.endswith('.pkl'):
                continue
            
            filepath = os.path.join(MODELS_DIR, filename)
            sensor_name = filename.replace('.pkl', '').replace('n', '')
            
            try:
                data = joblib.load(filepath)
                
                if 'model_class' in data:  # Нейросеть
                    key = f"{sensor_name}_nn"
                    self.models_cache[key] = {
                        'type': 'nn',
                        'data': data,
                        'auc': data.get('test_auc', 0),
                        'f1': data.get('test_f1', 0),
                        'model_name': data.get('model_class', 'Unknown')
                    }
                else:  # ML
                    key = f"{sensor_name}_ml"
                    metrics = data.get('test_metrics', data.get('metrics', {}))
                    self.models_cache[key] = {
                        'type': 'ml',
                        'data': data,
                        'auc': metrics.get('roc_auc', metrics.get('auc', 0)),
                        'f1': metrics.get('f1', 0),
                        'model_name': data.get('model_name', 'Unknown')
                    }
                    
            except Exception as e:
                print(f"❌ Ошибка загрузки {filename}: {e}")
        
        print(f"📊 Загружено моделей: {len(self.models_cache)}")
    
    def predict_ml(self, model_data, X):
        """Прогноз ML модели"""
        model = model_data['model']
        scaler = model_data['scaler']
        return model.predict_proba(scaler.transform(X))[:, 1]
    
    def predict_nn(self, model_data, X):
        """Прогноз нейросети"""
        state = model_data['model_state']
        input_dim = model_data['input_dim']
        model_class = model_data['model_class']
        scaler = model_data['scaler']
        
        ModelClass = NEURAL_MODELS.get(model_class)
        if ModelClass is None:
            raise ValueError(f"Неизвестная архитектура: {model_class}")
        
        model = ModelClass(input_dim)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        with torch.no_grad():
            return model(X_tensor).squeeze().cpu().numpy()
    
    def predict(self, sensor_name, df, use_model='best'):
        """
        Прогноз для датчика
        
        Args:
            sensor_name: имя датчика
            df: DataFrame с данными
            use_model: 'best', 'ml', 'nn'
        
        Returns:
            dict с прогнозом
        """
        ml_key = f"{sensor_name}_ml"
        nn_key = f"{sensor_name}_nn"
        
        # Выбор модели
        if use_model == 'ml':
            model_key = ml_key
        elif use_model == 'nn':
            model_key = nn_key
        else:  # best
            ml_model = self.models_cache.get(ml_key)
            nn_model = self.models_cache.get(nn_key)
            if ml_model and nn_model:
                model_key = ml_key if ml_model['auc'] >= nn_model['auc'] else nn_key
            else:
                model_key = ml_key or nn_key
        
        if model_key not in self.models_cache:
            raise ValueError(f"Модель для {sensor_name} не найдена")
        
        model_info = self.models_cache[model_key]
        model_data = model_info['data']
        
        # Создание признаков
        X = self.feature_creator.create_all_features(sensor_name, df)
        if X is None:
            raise ValueError("Не удалось создать признаки")
        
        # Прогноз
        if model_info['type'] == 'ml':
            predictions = self.predict_ml(model_data, X)
        else:
            predictions = self.predict_nn(model_data, X)
        
        threshold = model_data.get('threshold', 0.5)
        
        return {
            'predictions': predictions,
            'threshold': threshold,
            'model_used': model_info['model_name'],
            'model_type': model_info['type'],
            'auc': model_info['auc'],
            'f1': model_info['f1'],
            'working_ratio': float(np.mean(predictions >= threshold))
        }