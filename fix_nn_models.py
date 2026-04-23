# fix_nn_models.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
import os
import torch
import joblib

MODELS_DIR = "trained_models_best"

# ВАЖНО: Принудительно отключаем CUDA перед загрузкой
torch.cuda.is_available = lambda: False
import torch.serialization
# Патчим загрузку чтобы всегда использовать CPU
original_load = torch.load
def cpu_load(*args, **kwargs):
    kwargs['map_location'] = 'cpu'
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = cpu_load

print("🔧 Исправление нейросетевых моделей (CPU mode)...")

for filename in os.listdir(MODELS_DIR):
    if not filename.endswith('n.pkl'):
        continue
    
    filepath = os.path.join(MODELS_DIR, filename)
    print(f"   {filename}...", end=' ')
    
    try:
        # Загружаем (теперь всегда на CPU)
        data = joblib.load(filepath)
        
        if 'model_state' in data:
            # Все тензоры на CPU
            for key in data['model_state']:
                if isinstance(data['model_state'][key], torch.Tensor):
                    data['model_state'][key] = data['model_state'][key].cpu()
            
            # Сохраняем
            joblib.dump(data, filepath)
            print("✅ исправлена")
        else:
            print("⚠️ нет model_state")
            
    except Exception as e:
        print(f"❌ {e}")

print("\n✅ Готово! Запустите сервер: python main.py")