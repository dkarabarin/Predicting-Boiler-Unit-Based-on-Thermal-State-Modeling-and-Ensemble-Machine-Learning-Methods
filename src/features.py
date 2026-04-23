# src/features.py
import numpy as np
import pandas as pd
from .config import SENSORS, PHYSICS, STEPS_FORWARD, A_STEEL, U0_STEEL, GAMMA_STEEL, R_GAS

class FeatureCreator:
    """Создание признаков для прогнозирования"""
    
    def __init__(self, forecast_hours=24):
        self.forecast_hours = forecast_hours
        self.shift_steps = forecast_hours * 6
    
    def create_temporal_features(self, data):
        """Календарные признаки"""
        features = pd.DataFrame(index=data.index)
        idx = data.index
        
        features['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24)
        features['month_sin'] = np.sin(2 * np.pi * idx.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * idx.month / 12)
        features['dayofweek_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7)
        features['day_sin'] = np.sin(2 * np.pi * idx.dayofyear / 365)
        features['day_cos'] = np.cos(2 * np.pi * idx.dayofyear / 365)
        
        features['hour'] = idx.hour
        features['month'] = idx.month
        features['quarter'] = idx.quarter
        features['dayofweek'] = idx.dayofweek
        features['dayofyear'] = idx.dayofyear
        
        features['is_weekend'] = (idx.dayofweek >= 5).astype(int)
        features['is_heating_season'] = ((idx.month >= 10) | (idx.month <= 4)).astype(int)
        features['is_night'] = ((idx.hour >= 22) | (idx.hour <= 5)).astype(int)
        features['is_morning_peak'] = ((idx.hour >= 6) & (idx.hour <= 9)).astype(int)
        features['is_evening_peak'] = ((idx.hour >= 17) & (idx.hour <= 20)).astype(int)
        
        return features
    
    def create_degradation_features(self, sensor_name, data):
        """Признаки деградации со сдвигом"""
        features = pd.DataFrame(index=data.index)
        
        if sensor_name not in data.columns:
            return features
        
        temperature = data[sensor_name]
        surface_type = SENSORS.get(sensor_name, 'средние ширмы')
        factor = PHYSICS.get(surface_type, 1.0)
        
        # Термические напряжения
        thermal_stress = factor * np.maximum(temperature - 400, 0) / 50
        T_abs = temperature + 273.15
        T_p = A_STEEL * np.exp((U0_STEEL - GAMMA_STEEL * thermal_stress * 1e6) / (R_GAS * T_abs))
        
        # Деградация
        dt = 600
        degradation_rate = dt / (T_p + 1e-8)
        D_cumulative = degradation_rate.cumsum()
        
        shift = self.shift_steps
        
        features['D_cumulative'] = D_cumulative.shift(shift).fillna(0)
        features['degradation_speed'] = degradation_rate.rolling(144, min_periods=1).mean().shift(shift).fillna(0)
        features['D_ma_24h'] = D_cumulative.rolling(144, min_periods=1).mean().shift(shift).fillna(0)
        features['D_ma_168h'] = D_cumulative.rolling(1008, min_periods=1).mean().shift(shift).fillna(0)
        
        # Тренд деградации
        def calc_trend(series):
            if len(series) < 10:
                return 0
            return np.polyfit(np.arange(len(series)), series, 1)[0]
        
        features['D_trend_72h'] = D_cumulative.rolling(
            window=432, min_periods=10
        ).apply(calc_trend, raw=True).shift(shift).fillna(0)
        
        # Коэффициент загрязнения
        temp_norm = temperature.rolling(168, min_periods=1).mean().shift(shift)
        features['epsilon'] = ((temperature.shift(shift) - temp_norm) / (temp_norm + 1e-8)).fillna(0)
        
        # Накопленное тепловое воздействие (без температуры!)
        heat_exposure = np.cumsum(np.maximum(temperature - 300, 0)) / 1000000
        features['heat_exposure'] = pd.Series(heat_exposure, index=data.index).shift(shift).fillna(0)
        
        return features
    
    def create_statistical_features(self, sensor_name, data):
        """Статистические признаки"""
        features = pd.DataFrame(index=data.index)
        
        if sensor_name not in data.columns:
            return features
        
        temperature = data[sensor_name]
        shift = self.shift_steps
        
        seasonal_mean = temperature.groupby(temperature.index.month).transform('mean')
        seasonal_std = temperature.groupby(temperature.index.month).transform('std')
        features['seasonal_anomaly'] = ((temperature - seasonal_mean) / (seasonal_std + 1e-8)).shift(shift).fillna(0)
        
        # Паттерны работы (без температуры)
        working = (temperature > 200).astype(float)
        ws = pd.Series(working, index=data.index)
        
        features['working_ratio_7d'] = ws.rolling(1008, min_periods=1).sum().shift(shift).fillna(0).values / 1008
        features['working_ratio_30d'] = ws.rolling(4320, min_periods=1).sum().shift(shift).fillna(0).values / 4320
        
        return features
    
    def create_all_features(self, sensor_name, data):
        """Создание всех признаков"""
        if sensor_name not in data.columns:
            return None
        
        features_list = [
            self.create_temporal_features(data),
            self.create_degradation_features(sensor_name, data),
            self.create_statistical_features(sensor_name, data)
        ]
        
        all_features = pd.concat(features_list, axis=1)
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.ffill().bfill().fillna(0)
        
        for col in all_features.columns:
            if all_features[col].abs().max() > 1e6:
                all_features[col] = all_features[col].clip(-1e6, 1e6)
        
        return all_features