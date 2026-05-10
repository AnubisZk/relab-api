# RE Optimization Lab — FastAPI Backend
**ZSK Solutions** · Python Physics Engine + World Model API

## Dosya Yapısı

```
relab-api/
├── main.py              ← FastAPI app + tüm endpoint'ler
├── physics_engine.py    ← Python fizik motoru (JS ile birebir)
├── optimizer_scipy.py   ← SLSQP + Differential Evolution
├── world_model.py       ← scikit-learn MLPRegressor surrogate
├── requirements.txt     ← Python bağımlılıkları
├── Procfile             ← Railway start command
├── railway.toml         ← Railway deploy config
└── README.md
```

## Endpoint'ler

| Method | Path | Açıklama |
|--------|------|----------|
| GET  | `/api/health` | Sağlık kontrolü |
| POST | `/api/calculate` | Fizik motoru hesabı |
| POST | `/api/optimize` | SLSQP optimizasyon |
| POST | `/api/world-model/train` | ML model eğitimi |
| POST | `/api/world-model/predict` | Hızlı ML tahmini |
| POST | `/api/world-model/compare` | Physics vs WM karşılaştırma |
| POST | `/api/vr/sync` | Unity/VR sahne sync |

## Railway Deploy (Adım Adım)

### 1. GitHub'a Push Et
```bash
# Yeni repo: github.com → New → relab-api
cd relab-api
git init
git add .
git commit -m "RE Lab API v2 - Physics + World Model + Optimizer"
git remote add origin https://github.com/AnubisZk/relab-api.git
git push -u origin main
```

### 2. Railway'de Deploy Et
1. `railway.app` → New Project → Deploy from GitHub
2. Repo seç: `relab-api`
3. Railway otomatik algılar: Python → NIXPACKS
4. Deploy tıkla → 2-3 dakika
5. URL alırsın: `https://relab-api-production.up.railway.app`

### 3. Netlify'a API URL Ekle
Dashboard'da `app.js` içinde:
```javascript
const API_BASE = 'https://relab-api-production.up.railway.app';
```

### 4. World Model'i Eğit
```bash
curl -X POST https://your-api.railway.app/api/world-model/train \
  -H "Content-Type: application/json" \
  -d '{"n_samples": 5000}'
```
~2-3 dakika sürer. Status kontrol:
```bash
curl https://your-api.railway.app/api/model-status
```

## Test

```bash
# Health check
curl https://your-api.railway.app/api/health

# Hesaplama
curl -X POST https://your-api.railway.app/api/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "solar": {"panelArea": 5000, "irradiance": 850, "efficiency": 0.225, "tilt": 32},
    "wind":  {"windSpeed": 8.5, "turbineCount": 5},
    "battery": {"capacity": 1500, "demand": 4000}
  }'

# Optimizasyon
curl -X POST https://your-api.railway.app/api/optimize \
  -H "Content-Type: application/json" \
  -d '{"method": "slsqp", "n_restarts": 5}'
```

## Swagger UI
`https://your-api.railway.app/docs`

## Performans Karşılaştırması

| Yöntem | Süre | Kesinlik |
|--------|------|----------|
| JS Grid Search (312k combo) | ~30-60 sn | İyi |
| Python SLSQP (gradient) | ~0.5-2 sn | Çok iyi |
| Differential Evolution | ~5-15 sn | En iyi |
| World Model (ML) | <5 ms | ±2-5% |
