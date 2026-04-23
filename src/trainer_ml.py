# src/trainer_ml.py
import numpy as np
import time
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve
from .config import SENSORS, MODELS_DIR, STEPS_FORWARD
from .features import FeatureCreator
from .models_ml import get_ml_models

def train_ml_models(df, train_data, val2016, val2017, test_data):
    """Обучение ML моделей с GridSearch"""
    feature_creator = FeatureCreator()
    all_results = {}
    
    for sensor_name in SENSORS.keys():
        if sensor_name not in df.columns:
            continue
        
        print(f"\n📡 ML: {sensor_name} ({SENSORS[sensor_name]})")
        t_start = time.time()
        
        # Признаки
        X_train = feature_creator.create_all_features(sensor_name, train_data)
        X_val2016 = feature_creator.create_all_features(sensor_name, val2016)
        X_val2017 = feature_creator.create_all_features(sensor_name, val2017)
        X_test = feature_creator.create_all_features(sensor_name, test_data)
        
        if X_train is None:
            continue
        
        # Целевая
        y_train = train_data['is_working'].shift(-STEPS_FORWARD)
        y_val2016 = val2016['is_working'].shift(-STEPS_FORWARD)
        y_val2017 = val2017['is_working'].shift(-STEPS_FORWARD)
        y_test = test_data['is_working'].shift(-STEPS_FORWARD)
        
        # Обрезка и очистка
        X_train = X_train.iloc[:-STEPS_FORWARD]; X_val2016 = X_val2016.iloc[:-STEPS_FORWARD]
        X_val2017 = X_val2017.iloc[:-STEPS_FORWARD]; X_test = X_test.iloc[:-STEPS_FORWARD]
        y_train = y_train.iloc[:-STEPS_FORWARD]; y_val2016 = y_val2016.iloc[:-STEPS_FORWARD]
        y_val2017 = y_val2017.iloc[:-STEPS_FORWARD]; y_test = y_test.iloc[:-STEPS_FORWARD]
        
        vt = ~y_train.isna(); v6 = ~y_val2016.isna(); v7 = ~y_val2017.isna(); vtest = ~y_test.isna()
        X_train, y_train = X_train[vt], y_train[vt]
        X_val2016, y_val2016 = X_val2016[v6], y_val2016[v6]
        X_val2017, y_val2017 = X_val2017[v7], y_val2017[v7]
        X_test, y_test = X_test[vtest], y_test[vtest]
        
        # Масштабирование
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_v16_s = scaler.transform(X_val2016)
        X_v17_s = scaler.transform(X_val2017)
        X_te_s = scaler.transform(X_test)
        
        print(f"   Train: {len(X_tr_s):,} | Val: {len(X_v16_s):,}+{len(X_v17_s):,} | Test: {len(X_te_s):,}")
        
        # Обучение
        val_results = {}
        for name, config in get_ml_models().items():
            try:
                t0 = time.time()
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                grid = GridSearchCV(config['model'], config['params'], cv=cv, scoring='roc_auc', n_jobs=1)
                
                if 'fit_params' in config:
                    grid.fit(X_tr_s, y_train, **config['fit_params'])
                else:
                    grid.fit(X_tr_s, y_train)
                
                best_model = grid.best_estimator_
                
                # Валидация
                y_v16_pred = best_model.predict_proba(X_v16_s)[:, 1]
                y_v17_pred = best_model.predict_proba(X_v17_s)[:, 1]
                y_te_pred = best_model.predict_proba(X_te_s)[:, 1]
                
                p, r, t = precision_recall_curve(y_val2016, y_v16_pred)
                f1s = 2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-8)
                threshold = t[np.argmax(f1s)] if len(t) > 0 else 0.5
                
                roc_2016 = roc_auc_score(y_val2016, y_v16_pred)
                roc_2017 = roc_auc_score(y_val2017, y_v17_pred)
                test_auc = roc_auc_score(y_test, y_te_pred)
                test_f1 = f1_score(y_test, (y_te_pred >= threshold).astype(int))
                
                combined = test_auc * 0.5 + (roc_2016 + roc_2017) / 2 * 0.3 + test_f1 * 0.2
                
                print(f"   {name:20s}: Val16={roc_2016:.4f} Val17={roc_2017:.4f} Test AUC={test_auc:.4f} | {time.time()-t0:.0f}с")
                
                val_results[name] = {
                    'model': best_model, 'scaler': scaler, 'threshold': threshold,
                    'test_auc': test_auc, 'test_f1': test_f1, 'combined': combined
                }
                
            except Exception as e:
                print(f"   ❌ {name}: {str(e)[:80]}")
        
        # Сохранение лучшей
        if val_results:
            best_name = max(val_results, key=lambda x: val_results[x]['combined'])
            best = val_results[best_name]
            
            model_data = {
                'model': best['model'], 'scaler': best['scaler'],
                'threshold': best['threshold'], 'model_name': best_name,
                'metrics': {'auc': best['test_auc'], 'f1': best['test_f1']}
            }
            
            path = os.path.join(MODELS_DIR, f"{sensor_name}.pkl")
            joblib.dump(model_data, path)
            print(f"   💾 {best_name} AUC={best['test_auc']:.4f} | {time.time()-t_start:.0f}с")
            
            all_results[sensor_name] = model_data
    
    return all_results