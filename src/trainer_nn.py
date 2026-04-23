# src/trainer_nn.py
import numpy as np
import time
import joblib
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

from .config import SENSORS, MODELS_DIR, STEPS_FORWARD
from .features import FeatureCreator
from .models_nn import NEURAL_MODELS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    def __init__(self, model, lr=0.001, weight_decay=1e-4):
        self.model = model.to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=7)
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X_batch).squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    def evaluate(self, loader):
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                preds.extend(self.model(X_batch.to(device)).squeeze().cpu().numpy())
                targets.extend(y_batch.numpy())
        return np.array(preds), np.array(targets)
    
    def fit(self, train_loader, val_loader, epochs=100, patience=15):
        best_roc = 0
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.train_epoch(train_loader)
            val_pred, val_true = self.evaluate(val_loader)
            val_roc = roc_auc_score(val_true, val_pred)
            self.scheduler.step(val_roc)
            
            if val_roc > best_roc:
                best_roc = val_roc
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        if best_state:
            self.model.load_state_dict(best_state)
        return best_roc


def train_nn_models(df, train_data, val2016, val2017, test_data):
    """Обучение нейросетей"""
    feature_creator = FeatureCreator()
    all_val = pd.concat([val2016, val2017])
    nn_results = {}
    
    for sensor_name in SENSORS.keys():
        if sensor_name not in df.columns:
            continue
        
        print(f"\n🧠 NN: {sensor_name} ({SENSORS[sensor_name]})")
        t_start = time.time()
        
        # Признаки
        X_train = feature_creator.create_all_features(sensor_name, train_data)
        X_val = feature_creator.create_all_features(sensor_name, all_val)
        X_test = feature_creator.create_all_features(sensor_name, test_data)
        
        if X_train is None:
            continue
        
        # Целевая
        y_train = train_data['is_working'].shift(-STEPS_FORWARD).iloc[:-STEPS_FORWARD]
        y_val = all_val['is_working'].shift(-STEPS_FORWARD).iloc[:-STEPS_FORWARD]
        y_test = test_data['is_working'].shift(-STEPS_FORWARD).iloc[:-STEPS_FORWARD]
        
        X_train = X_train.iloc[:len(y_train)]
        X_val = X_val.iloc[:len(y_val)]
        X_test = X_test.iloc[:len(y_test)]
        
        vt = ~y_train.isna(); vv = ~y_val.isna(); vtest = ~y_test.isna()
        X_train, y_train = X_train[vt], y_train[vt]
        X_val, y_val = X_val[vv], y_val[vv]
        X_test, y_test = X_test[vtest], y_test[vtest]
        
        # Масштабирование
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_te_s = scaler.transform(X_test)
        
        input_dim = X_tr_s.shape[1]
        print(f"   Input dim: {input_dim}, Train: {len(X_tr_s):,}")
        
        # Тензоры
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_train.values)),
            batch_size=256, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val.values)),
            batch_size=256, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_te_s), torch.FloatTensor(y_test.values)),
            batch_size=256, shuffle=False
        )
        
        best_score = -1
        best_nn_data = None
        
        for name, ModelClass in NEURAL_MODELS.items():
            try:
                t0 = time.time()
                model = ModelClass(input_dim)
                trainer = Trainer(model)
                val_roc = trainer.fit(train_loader, val_loader)
                
                y_val_pred, y_val_true = trainer.evaluate(val_loader)
                y_test_pred, y_test_true = trainer.evaluate(test_loader)
                
                p, r, t = precision_recall_curve(y_val_true, y_val_pred)
                f1s = 2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-8)
                threshold = t[np.argmax(f1s)] if len(t) > 0 else 0.5
                
                test_auc = roc_auc_score(y_test_true, y_test_pred)
                test_f1 = f1_score(y_test_true, (y_test_pred >= threshold).astype(int))
                score = test_auc * 0.6 + test_f1 * 0.4
                
                print(f"   {name:15s}: Val ROC={val_roc:.4f} Test AUC={test_auc:.4f} F1={test_f1:.4f} | {time.time()-t0:.0f}с")
                
                if score > best_score:
                    best_score = score
                    best_nn_data = {
                        'model_state': copy.deepcopy(model.state_dict()),
                        'model_class': name,
                        'input_dim': input_dim,
                        'scaler': scaler,
                        'threshold': threshold,
                        'test_auc': test_auc,
                        'test_f1': test_f1
                    }
                
            except Exception as e:
                print(f"   ❌ {name}: {str(e)[:80]}")
        
        # Сохранение
        if best_nn_data:
            path = os.path.join(MODELS_DIR, f"{sensor_name}n.pkl")
            joblib.dump(best_nn_data, path)
            print(f"   💾 {best_nn_data['model_class']} AUC={best_nn_data['test_auc']:.4f} | {time.time()-t_start:.0f}с")
            nn_results[sensor_name] = best_nn_data
        
        torch.cuda.empty_cache()
    
    return nn_results