# src/models_ml.py
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import torch

gpu_available = torch.cuda.is_available()

def get_ml_models():
    """ML модели с GridSearch"""
    models = {}
    
    models['RandomForest'] = {
        'model': RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight='balanced',
            min_samples_leaf=20, min_samples_split=50, max_features='sqrt'
        ),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15],
            'min_samples_leaf': [20, 50]
        }
    }
    
    models['LightGBM'] = {
        'model': lgb.LGBMClassifier(
            random_state=42, n_jobs=-1, verbose=-1,
            device='gpu' if gpu_available else 'cpu',
            learning_rate=0.1, n_estimators=150
        ),
        'params': {
            'num_leaves': [31, 63],
            'learning_rate': [0.05, 0.1],
            'max_depth': [5, 7],
            'min_child_samples': [30, 50]
        },
        'fit_params': {'eval_metric': 'logloss'}
    }
    
    models['XGBoost'] = {
        'model': xgb.XGBClassifier(
            random_state=42, n_jobs=-1, verbosity=0,
            tree_method='hist',
            device='cuda' if gpu_available else 'cpu',
            eval_metric='logloss',
            learning_rate=0.1, n_estimators=150
        ),
        'params': {
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0]
        }
    }
    
    models['GradientBoosting'] = {
        'model': GradientBoostingClassifier(
            random_state=42, subsample=0.8, max_features='sqrt',
            min_samples_leaf=20, min_samples_split=50
        ),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1]
        }
    }
    
    models['LogisticRegression'] = {
        'model': LogisticRegression(
            random_state=42, max_iter=1000, n_jobs=-1,
            class_weight='balanced', solver='liblinear'
        ),
        'params': {
            'C': [0.01, 0.1, 1.0],
            'penalty': ['l2']
        }
    }
    
    models['ExtraTrees'] = {
        'model': ExtraTreesClassifier(
            random_state=42, n_jobs=-1, class_weight='balanced',
            min_samples_leaf=20, min_samples_split=50, max_features='sqrt'
        ),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15],
            'min_samples_leaf': [20, 50]
        }
    }
    
    return models