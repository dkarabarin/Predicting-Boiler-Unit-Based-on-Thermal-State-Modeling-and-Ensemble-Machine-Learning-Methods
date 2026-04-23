# main.py - FastAPI сервис прогнозирования ТЭС
import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================
# НАСТРОЙКА
# ============================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI(title="🔥 Прогнозирование ТЭС", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ============================================
# КОНФИГУРАЦИЯ
# ============================================
SENSORS_CONFIG = {
    '10HAH01CT103': 'Средние ширмы 1', '10HAH01CT102': 'Средние ширмы 2',
    '10HAH12CT101': 'Ширмы 1', '10HAH12CT104': 'Ширмы 2',
    '10HAH12CT110': 'Пароперегреватель 1', '10HAH12CT108': 'Пароперегреватель 2',
    '10HAH12CT106': 'Пароперегреватель 3', '10HAH11CT114': 'Пароперегреватель 4',
    '10HAH11CT113': 'Пароперегреватель 5', '10HAH12CT116': 'Ширмы 3',
    '10HAH12CT117': 'Ширмы 4'
}

MODELS_DIR = "trained_models_best"
STEPS_FORWARD = 24 * 6  # 144

# ============================================
# НЕЙРОСЕТЕВЫЕ АРХИТЕКТУРЫ
# ============================================
class SimpleDNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

class WideDeepNet(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        self.wide = nn.Linear(input_dim, 16)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.combined = nn.Sequential(
            nn.Linear(48, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.combined(torch.cat([self.wide(x), self.deep(x)], dim=1))

class AttentionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout_rate, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x, _ = self.attention(x, x, x)
        return self.fc(self.norm(x.squeeze(1)))

NEURAL_MODELS = {'SimpleDNN': SimpleDNN, 'WideDeepNet': WideDeepNet, 'AttentionNet': AttentionNet}

# ============================================
# ПРИЗНАКИ (копия из ноутбука)
# ============================================
A_STEEL, U0_STEEL, GAMMA_STEEL, R_GAS = 1.2e-12, 480000, 0.35, 8.314

class FeatureCreator:
    def __init__(self, forecast_hours=24):
        self.shift_steps = forecast_hours * 6

    def create_features(self, data, sensor_name):
        features = pd.DataFrame(index=data.index)
        idx = data.index
        shift = self.shift_steps

        # Темпоральные
        features['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24)
        features['month_sin'] = np.sin(2 * np.pi * idx.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * idx.month / 12)
        features['dayofweek_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7)
        features['hour'] = idx.hour
        features['month'] = idx.month
        features['quarter'] = idx.quarter
        features['dayofweek'] = idx.dayofweek
        features['is_weekend'] = (idx.dayofweek >= 5).astype(int)
        features['is_heating_season'] = ((idx.month >= 10) | (idx.month <= 4)).astype(int)
        features['is_night'] = ((idx.hour >= 22) | (idx.hour <= 5)).astype(int)
        features['is_morning_peak'] = ((idx.hour >= 6) & (idx.hour <= 9)).astype(int)
        features['is_evening_peak'] = ((idx.hour >= 17) & (idx.hour <= 20)).astype(int)

        # Деградация (нули для прогноза)
        features['D_cumulative'] = 0
        features['degradation_speed'] = 0
        features['D_ma_24h'] = 0
        features['D_ma_168h'] = 0
        features['D_trend_72h'] = 0
        features['epsilon'] = 0

        # Статистические (нули для прогноза)
        features['seasonal_anomaly'] = 0

        # Если есть температура
        if sensor_name in data.columns:
            temp = data[sensor_name].values
            thermal_stress = 1.0 * np.maximum(temp - 400, 0) / 50
            T_abs = temp + 273.15
            T_p = A_STEEL * np.exp((U0_STEEL - GAMMA_STEEL * thermal_stress * 1e6) / (R_GAS * T_abs))
            degradation_rate = 600 / (T_p + 1e-8)
            D_cumulative = degradation_rate.cumsum()
            features['D_cumulative'] = np.roll(D_cumulative, shift)
            features['D_cumulative'][:shift] = 0
            features['degradation_speed'] = np.roll(degradation_rate, shift)
            features['degradation_speed'][:shift] = 0

        return features.fillna(0)

feature_creator = FeatureCreator()

# ============================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================
models_cache = {}
uploaded_data: pd.DataFrame = None

# ============================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================
@app.on_event("startup")
async def startup():
    global models_cache
    if not os.path.exists(MODELS_DIR):
        logger.warning(f"⚠️ Папка {MODELS_DIR} не найдена")
        return

    for filename in os.listdir(MODELS_DIR):
        if not filename.endswith('.pkl'):
            continue
        filepath = os.path.join(MODELS_DIR, filename)
        sensor_name = filename.replace('.pkl', '').replace('n', '')
        try:
            data = joblib.load(filepath)
            if 'model_class' in data:  # Нейросеть
                if 'model_state' in data:
                    data['model_state'] = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in data['model_state'].items()}
                model_info = {'type': 'nn', 'data': data, 'auc': data.get('test_auc', 0), 'f1': data.get('test_f1', 0), 'model_name': data.get('model_class', 'Unknown')}
                key = f"{sensor_name}_nn"
            else:  # ML
                metrics = data.get('test_metrics', data.get('metrics', {}))
                model_info = {'type': 'ml', 'data': data, 'auc': metrics.get('roc_auc', metrics.get('auc', 0)), 'f1': metrics.get('f1', 0), 'model_name': data.get('model_name', 'Unknown')}
                key = f"{sensor_name}_ml"
            models_cache[key] = model_info
            logger.info(f"✅ {key}: {model_info['model_name']} (AUC={model_info['auc']:.4f})")
        except Exception as e:
            logger.error(f"❌ {filename}: {e}")
    logger.info(f"📊 Загружено моделей: {len(models_cache)}")

# ============================================
# ПРОГНОЗИРОВАНИЕ
# ============================================
def predict_ml(model_data, X):
    model, scaler = model_data['model'], model_data['scaler']
    # Оставляем только признаки которые были при обучении
    expected = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X.columns
    X = X.reindex(columns=expected, fill_value=0)
    return model.predict_proba(scaler.transform(X))[:, 1]

def predict_nn(model_data, X):
    state = model_data['model_state']
    ModelClass = NEURAL_MODELS.get(model_data['model_class'], SimpleDNN)
    model = ModelClass(model_data['input_dim'])
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    X_s = model_data['scaler'].transform(X)
    with torch.no_grad():
        return model(torch.FloatTensor(X_s).to(DEVICE)).squeeze().cpu().numpy()

def get_model(sensor_name, use_model='best'):
    ml_key, nn_key = f"{sensor_name}_ml", f"{sensor_name}_nn"
    ml_m, nn_m = models_cache.get(ml_key), models_cache.get(nn_key)
    if use_model == 'ml': key = ml_key
    elif use_model == 'nn': key = nn_key
    else:
        if ml_m and nn_m: key = ml_key if ml_m['auc'] >= nn_m['auc'] else nn_key
        else: key = ml_key or nn_key
    if not key or key not in models_cache:
        raise ValueError(f"Модель для {sensor_name} не найдена")
    return key, models_cache[key]

# ============================================
# Pydantic модели
# ============================================
class PredictionRequest(BaseModel):
    sensor_name: str
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    use_model: str = "best"

class PredictionResponse(BaseModel):
    sensor_name: str
    sensor_label: str
    model_used: str
    model_type: str
    auc: float
    f1: float
    threshold: float
    forecast_dates: list
    predictions: list
    working_ratio: float
    working_hours: float
    downtime_hours: float
    monthly_summary: list

# ============================================
# ВЕБ-ИНТЕРФЕЙС
# ============================================
@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>🔥 Прогнозирование ТЭС</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px}.container{max-width:1400px;margin:0 auto;background:#fff;border-radius:20px;padding:30px;box-shadow:0 20px 60px rgba(0,0,0,.3)}.header{text-align:center;margin-bottom:30px}.header h1{font-size:2.2em;color:#333}.header p{color:#666;margin-top:10px}.steps{display:flex;justify-content:space-between;margin:20px 0;gap:10px}.step{flex:1;text-align:center;padding:12px;background:#f8f9fa;border-radius:10px}.step .num{width:35px;height:35px;border-radius:50%;background:#dee2e6;display:flex;align-items:center;justify-content:center;margin:0 auto 8px;font-weight:700}.step.active{background:#667eea;color:#fff}.step.active .num{background:#fff;color:#667eea}.step.completed{background:#28a745;color:#fff}.step.completed .num{background:#fff;color:#28a745}.card{background:#f8f9fa;padding:20px;border-radius:12px;margin:15px 0}.upload-zone{border:3px dashed #667eea;border-radius:15px;padding:30px;text-align:center;cursor:pointer;background:#fff;transition:.3s}.upload-zone:hover{background:#f0f0ff}.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin:15px 0}.control-group label{display:block;font-weight:600;margin-bottom:5px;color:#495057}.control-group select,.control-group input{width:100%;padding:10px;border:2px solid #dee2e6;border-radius:8px;font-size:14px}.control-group select:focus,.control-group input:focus{outline:0;border-color:#667eea}.btn{padding:12px 30px;border:0;border-radius:8px;font-size:16px;font-weight:600;cursor:pointer;transition:.3s}.btn-primary{background:#667eea;color:#fff}.btn-success{background:#28a745;color:#fff}.btn:hover{transform:translateY(-2px);box-shadow:0 5px 15px rgba(0,0,0,.2)}.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin:20px 0}.stat-card{background:#fff;padding:20px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.1);border-left:4px solid #667eea}.stat-card.nn{border-left-color:#28a745}.stat-card h3{font-size:12px;color:#6c757d;margin-bottom:8px;text-transform:uppercase}.stat-card .value{font-size:22px;font-weight:700;color:#495057}.chart-container{background:#fff;padding:20px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.1);margin:20px 0}#plot{width:100%;height:400px}.alert{padding:12px;border-radius:8px;margin:10px 0}.alert-success{background:#d4edda;color:#155724}.alert-danger{background:#f8d7da;color:#721c24}.alert-warning{background:#fff3cd;color:#856404}.loading{display:none;text-align:center;padding:20px}.loading.active{display:block}.spinner{border:4px solid #f3f3f3;border-top:4px solid #667eea;border-radius:50%;width:40px;height:40px;animation:spin 1s linear infinite;margin:0 auto}@keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}table{width:100%;border-collapse:collapse;margin:10px 0}th,td{padding:8px;text-align:left;border-bottom:1px solid #dee2e6}th{background:#f8f9fa}.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;color:#fff}.badge-ml{background:#667eea}.badge-nn{background:#28a745}</style></head>
<body><div class="container"><div class="header"><h1>🔥 Прогнозирование остановов ТЭС</h1><p>ML + Нейросети | Прогноз на год | Загрузите CSV → Датчик → Прогноз</p></div>
<div class="steps"><div class="step" id="step1"><div class="num">1</div>Загрузка CSV</div><div class="step" id="step2"><div class="num">2</div>Датчик</div><div class="step" id="step3"><div class="num">3</div>Прогноз</div><div class="step" id="step4"><div class="num">4</div>Результат</div></div>
<div class="card"><h2>📁 Шаг 1: Загрузите данные</h2><div class="upload-zone" onclick="document.getElementById('fileInput').click()"><h3>📂 Нажмите или перетащите CSV</h3><p style="color:#6c757d;margin-top:8px">Date_Time + датчики температуры</p><input type="file" id="fileInput" accept=".csv" style="display:none" onchange="uploadFile(this.files[0])"></div><p id="fileStatus" style="margin-top:10px;color:#6c757d">Файл не загружен</p></div>
<div class="card"><h2>⚙️ Шаг 2: Настройки</h2><div class="controls"><div class="control-group"><label>🎯 Датчик:</label><select id="sensorSelect"><option value="">Выберите...</option></select></div><div class="control-group"><label>🤖 Модель:</label><select id="modelType"><option value="best">🏆 Лучшая</option><option value="ml">🤖 ML</option><option value="nn">🧠 Нейросеть</option></select></div><div class="control-group"><label>📅 Начало:</label><input type="date" id="startDate" value="2025-01-01"></div><div class="control-group"><label>📅 Конец:</label><input type="date" id="endDate" value="2025-12-31"></div></div><button class="btn btn-success" onclick="makePrediction()" style="width:100%;font-size:18px;padding:15px">🚀 Выполнить прогноз</button></div>
<div class="loading" id="loading"><div class="spinner"></div><p>Выполняется прогноз...</p></div><div id="message"></div><div id="results" style="display:none"><div class="stats-grid" id="statsGrid"></div><div class="chart-container"><div id="plot"></div></div><div class="card" id="monthlySection" style="display:none"><h2>📅 Помесячная сводка</h2><div id="monthlyTable"></div></div></div></div>
<script>let fileOk=false;async function loadSensors(){const r=await fetch('/api/sensors');const s=await r.json();const sel=document.getElementById('sensorSelect');sel.innerHTML='<option value="">Выберите...</option>';for(const[id,name]of Object.entries(s))sel.innerHTML+=`<option value="${id}">${name}</option>`}async function uploadFile(file){if(!file)return;document.getElementById('fileStatus').innerHTML='⏳ Загрузка...';const fd=new FormData();fd.append('file',file);try{const r=await fetch('/api/upload',{method:'POST',body:fd});if(r.ok){const d=await r.json();fileOk=true;document.getElementById('fileStatus').innerHTML=`✅ ${d.filename} | ${d.rows.toLocaleString()} записей`;document.getElementById('fileStatus').style.color='#28a745';document.getElementById('step1').classList.add('completed');const sel=document.getElementById('sensorSelect');sel.innerHTML='<option value="">Выберите...</option>';if(d.available_sensors)d.available_sensors.forEach(s=>sel.innerHTML+=`<option value="${s}">${s}</option>`)}}catch(e){document.getElementById('fileStatus').innerHTML='❌ Ошибка';document.getElementById('fileStatus').style.color='#dc3545'}}document.querySelector('.upload-zone').addEventListener('dragover',e=>{e.preventDefault();e.target.style.background='#f0f0ff'});document.querySelector('.upload-zone').addEventListener('dragleave',e=>e.target.style.background='');document.querySelector('.upload-zone').addEventListener('drop',e=>{e.preventDefault();e.target.style.background='';const f=e.dataTransfer.files[0];if(f&&f.name.endsWith('.csv'))uploadFile(f)});async function makePrediction(){const sensor=document.getElementById('sensorSelect').value;if(!sensor)return showMsg('warning','⚠️ Выберите датчик!');if(!fileOk)return showMsg('warning','⚠️ Загрузите CSV!');document.getElementById('loading').classList.add('active');document.getElementById('results').style.display='none';try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sensor_name:sensor,start_date:document.getElementById('startDate').value,end_date:document.getElementById('endDate').value,use_model:document.getElementById('modelType').value})});if(r.ok){displayResults(await r.json());['step2','step3','step4'].forEach(s=>document.getElementById(s).classList.add('completed'))}else{const e=await r.json();showMsg('danger','❌ '+e.detail)}}catch(e){showMsg('danger','❌ '+e.message)}finally{document.getElementById('loading').classList.remove('active')}}function displayResults(r){document.getElementById('results').style.display='block';const badge=r.model_type==='nn'?'<span class="badge badge-nn">🧠 NN</span>':'<span class="badge badge-ml">🤖 ML</span>';document.getElementById('statsGrid').innerHTML=`<div class="stat-card ${r.model_type==='nn'?'nn':''}"><h3>Модель ${badge}</h3><div class="value" style="font-size:14px">${r.model_used}</div><small>AUC:${r.auc?.toFixed(3)||'N/A'} | F1:${r.f1?.toFixed(3)||'N/A'}</small></div><div class="stat-card"><h3>Прогноз работы</h3><div class="value">${(r.working_ratio*100).toFixed(1)}%</div></div><div class="stat-card"><h3>Часов работы</h3><div class="value">${r.working_hours.toFixed(0)} ч</div></div><div class="stat-card"><h3>Часов простоя</h3><div class="value">${r.downtime_hours.toFixed(0)} ч</div></div>`;Plotly.newPlot('plot',[{x:r.forecast_dates,y:r.predictions,type:'scatter',mode:'lines',name:'Вероятность работы',line:{color:'#667eea',width:2},fill:'tozeroy',fillcolor:'rgba(102,126,234,.15)'},{x:r.forecast_dates,y:Array(r.forecast_dates.length).fill(r.threshold),type:'scatter',mode:'lines',name:'Порог',line:{color:'#dc3545',width:2,dash:'dash'}}],{title:`Прогноз: ${r.sensor_label}`,xaxis:{title:'Дата'},yaxis:{title:'Вероятность',range:[0,1]},hovermode:'x unified'});if(r.monthly_summary?.length){document.getElementById('monthlySection').style.display='block';let h='<table><tr><th>Месяц</th><th>Работа%</th><th>Часы работы</th><th>Часы простоя</th></tr>';r.monthly_summary.forEach(m=>h+=`<tr><td><b>${m.month}</b></td><td>${(m.working_ratio*100).toFixed(1)}%</td><td>${m.working_hours.toFixed(0)}</td><td>${m.downtime_hours.toFixed(0)}</td></tr>`);h+='</table>';document.getElementById('monthlyTable').innerHTML=h}}function showMsg(t,txt){const c={success:'alert-success',danger:'alert-danger',warning:'alert-warning'};document.getElementById('message').innerHTML=`<div class="alert ${c[t]||'alert-warning'}">${txt}</div>`;setTimeout(()=>document.getElementById('message').innerHTML='',5000)}loadSensors()</script></body></html>"""

# ============================================
# API
# ============================================
@app.get("/api/sensors")
async def get_sensors(): return SENSORS_CONFIG

@app.get("/api/models-info")
async def models_info():
    return {k: {'type': v['type'], 'model_name': v['model_name'], 'auc': round(v['auc'], 4), 'f1': round(v['f1'], 4)} for k, v in models_cache.items()}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    global uploaded_data
    try:
        df = pd.read_csv(file.file, parse_dates=['Date_Time']); df.set_index('Date_Time', inplace=True); df.sort_index(inplace=True)
        uploaded_data = df
        avail = [s for s in SENSORS_CONFIG if s in df.columns]
        return {"success": True, "filename": file.filename, "rows": len(df), "sensors": len(avail), "available_sensors": avail, "date_range": {"start": str(df.index.min()), "end": str(df.index.max())}}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    global uploaded_data
    if uploaded_data is None: raise HTTPException(400, "Сначала загрузите CSV!")
    if request.sensor_name not in SENSORS_CONFIG: raise HTTPException(400, f"Датчик {request.sensor_name} не найден")
    if request.sensor_name not in uploaded_data.columns: raise HTTPException(400, f"Датчик отсутствует в данных")

    try:
        sensor_data = uploaded_data[request.sensor_name].dropna()
        start_date = pd.to_datetime(request.start_date); end_date = pd.to_datetime(request.end_date)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Данные для прогноза
        monthly_means = sensor_data.groupby(sensor_data.index.month).mean()
        forecast_df = pd.DataFrame(index=dates)
        for d in dates: forecast_df.loc[d, request.sensor_name] = monthly_means.get(d.month, sensor_data.mean())

        # Признаки
        X = feature_creator.create_features(forecast_df, request.sensor_name)

        # Модель
        model_key, model_info = get_model(request.sensor_name, request.use_model)
        model_data = model_info['data']

        # Прогноз
        if model_info['type'] == 'ml':
            predictions = predict_ml(model_data, X)
        else:
            predictions = predict_nn(model_data, X)

        threshold = model_data.get('threshold', 0.5)
        working_ratio = float(np.mean(predictions >= threshold))
        total_hours = len(dates) * 24
        working_hours = working_ratio * total_hours
        downtime_hours = total_hours - working_hours

        monthly_summary = []
        for month in range(1, 13):
            mask = [d.month == month for d in dates]; mp = predictions[mask]
            if len(mp) > 0:
                mr = float(np.mean(mp >= threshold)); mh = len(mp) * 24
                monthly_summary.append({'month': f'{month:02d}', 'working_ratio': mr, 'working_hours': mr * mh, 'downtime_hours': (1 - mr) * mh})

        return {
            'sensor_name': request.sensor_name,
            'sensor_label': SENSORS_CONFIG.get(request.sensor_name, request.sensor_name),
            'model_used': model_info['model_name'], 'model_type': model_info['type'],
            'auc': float(model_info['auc'] or 0), 'f1': float(model_info['f1'] or 0),
            'threshold': float(threshold),
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in dates],
            'predictions': [float(p) for p in predictions],
            'working_ratio': working_ratio, 'working_hours': float(working_hours),
            'downtime_hours': float(downtime_hours), 'monthly_summary': monthly_summary
        }
    except ValueError as e: raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Predict error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": len(models_cache), "data_loaded": uploaded_data is not None, "device": str(DEVICE), "gpu": torch.cuda.is_available(), "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Запуск на {DEVICE}")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)