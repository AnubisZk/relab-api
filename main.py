"""
================================================================
RE OPTIMIZATION LAB — FastAPI Backend
ZSK Solutions · main.py
----------------------------------------------------------------
Endpoints:
  POST /api/calculate          Physics engine calculation
  POST /api/optimize           SLSQP gradient optimization
  POST /api/optimize/global    Differential evolution (global)
  POST /api/world-model/train  Train surrogate ML model
  POST /api/world-model/predict Fast ML prediction
  POST /api/vr/sync            Unity/VR scene sync
  GET  /api/health             Health check
  GET  /api/model-status       World model status
================================================================
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import time
from contextlib import asynccontextmanager

from physics_engine import (
    SolarParams, WindParams, OffshoreParams, BatteryParams, CostParams,
    calculate_all
)
from optimizer_scipy import run_slsqp, run_differential_evolution
from world_model import world_model

# ── AUTO TRAIN ON STARTUP ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Train world model on startup if not already trained
    if not world_model.trained:
        import threading
        def auto_train():
            print("[Startup] World model not found — auto-training with 10,000 samples...")
            world_model.train(n_samples=10000)
        threading.Thread(target=auto_train, daemon=True).start()
    yield

# ── APP ─────────────────────────────────────────────────────
app = FastAPI(
    title="RE Optimization Lab API",
    description="Renewable Energy Physics Engine + World Model + Optimizer · ZSK Solutions",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── REQUEST / RESPONSE MODELS ────────────────────────────────

class CalculateRequest(BaseModel):
    solar: dict = Field(default_factory=dict)
    wind:  dict = Field(default_factory=dict)
    offshore: dict = Field(default_factory=dict)
    battery:  dict = Field(default_factory=dict)
    cost:     dict = Field(default_factory=dict)


class OptimizeRequest(BaseModel):
    solar:    dict = Field(default_factory=dict)
    wind:     dict = Field(default_factory=dict)
    offshore: dict = Field(default_factory=dict)
    battery:  dict = Field(default_factory=dict)
    cost:     dict = Field(default_factory=dict)
    method:   str  = "slsqp"   # "slsqp" | "de"
    n_restarts: int = 5


class WorldModelRequest(BaseModel):
    inputs: dict = Field(default_factory=dict)


class VRSyncRequest(BaseModel):
    units: List[dict] = Field(default_factory=list)   # [{type, x, y, ...}]
    sliders: dict     = Field(default_factory=dict)
    scene_id: Optional[str] = None


class TrainRequest(BaseModel):
    n_samples: int = 3000


# ── HELPERS ─────────────────────────────────────────────────

def dict_to_solar(d: dict) -> SolarParams:
    return SolarParams(**{k: v for k, v in d.items() if hasattr(SolarParams, k) or k in SolarParams.__dataclass_fields__})

def dict_to_params(req: CalculateRequest):
    def safe(cls, d):
        fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in fields})
    solar_p   = safe(SolarParams,   req.solar)    if req.solar    else SolarParams()
    wind_p    = safe(WindParams,    req.wind)     if req.wind     else WindParams()
    offshore_p= safe(OffshoreParams,req.offshore) if req.offshore else OffshoreParams()
    battery_p = safe(BatteryParams, req.battery)  if req.battery  else BatteryParams()
    cost_p    = safe(CostParams,    req.cost)     if req.cost     else CostParams()
    return solar_p, wind_p, offshore_p, battery_p, cost_p


# ── ENDPOINTS ────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "world_model_trained": world_model.trained,
        "world_model_confidence": world_model.metrics.get('confidence', None),
    }


@app.get("/api/model-status")
def model_status():
    return {
        "trained":    world_model.trained,
        "metrics":    world_model.metrics,
    }


@app.post("/api/calculate")
def calculate(req: CalculateRequest):
    """
    Full physics engine calculation.
    Returns: solar, wind, offshore, battery, objective, violations
    """
    t0 = time.time()
    try:
        solar_p, wind_p, offshore_p, battery_p, cost_p = dict_to_params(req)
        result = calculate_all(solar_p, wind_p, offshore_p, battery_p, cost_p)
        result['_meta'] = {'elapsed_ms': round((time.time()-t0)*1000, 1), 'engine': 'physics'}
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    """
    Gradient-based optimization (SLSQP or Differential Evolution).
    Much faster than JS Grid Search.
    """
    t0 = time.time()
    try:
        def safe(cls, d):
            fields = cls.__dataclass_fields__.keys()
            return cls(**{k: v for k, v in d.items() if k in fields})

        solar_p    = safe(SolarParams,   req.solar)    if req.solar    else SolarParams()
        wind_p     = safe(WindParams,    req.wind)     if req.wind     else WindParams()
        offshore_p = safe(OffshoreParams,req.offshore) if req.offshore else OffshoreParams()
        battery_p  = safe(BatteryParams, req.battery)  if req.battery  else BatteryParams()
        cost_p     = safe(CostParams,    req.cost)     if req.cost     else CostParams()

        if req.method == 'de':
            result = run_differential_evolution(solar_p, wind_p, offshore_p, battery_p, cost_p)
        else:
            result = run_slsqp(solar_p, wind_p, offshore_p, battery_p, cost_p, n_restarts=req.n_restarts)

        result['_meta'] = {'elapsed_ms': round((time.time()-t0)*1000, 1), 'method': req.method}
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/world-model/train")
async def train_world_model(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Train surrogate ML model on physics engine simulation data.
    Runs in background — check /api/model-status for completion.
    """
    if world_model.trained:
        # Allow retraining
        pass

    def do_train():
        world_model.train(n_samples=req.n_samples)

    background_tasks.add_task(do_train)
    return {
        "message": f"Training started with {req.n_samples} samples",
        "status":  "training",
        "note":    "Poll /api/model-status to check completion",
    }


@app.post("/api/world-model/predict")
def wm_predict(req: WorldModelRequest):
    """
    Fast surrogate prediction — also computes surprise score.
    Requires model to be trained first.
    """
    t0 = time.time()
    if not world_model.trained:
        raise HTTPException(status_code=503, detail="World model not trained yet. POST /api/world-model/train first.")

    try:
        prediction = world_model.predict(req.inputs)

        # Also run physics for surprise computation if inputs complete enough
        surprise = None
        try:
            def safe(cls, d):
                fields = cls.__dataclass_fields__.keys()
                return cls(**{k: v for k, v in d.items() if k in fields})
            i = req.inputs
            solar_p    = SolarParams(tilt=i.get('panelTilt',32), azimuth=i.get('panelAzimuth',180), panelArea=i.get('panelArea',5000), irradiance=i.get('irradiance',850), cellTemp=i.get('cellTemp',35), dustLoss=i.get('dustLoss',0.15), shadeLoss=i.get('shadeLoss',0.05))
            wind_p     = WindParams(windSpeed=i.get('windSpeed',8.5), yawAngle=i.get('yawAngle',0), turbineSpacing=i.get('turbineSpacing',7), rotorRadius=i.get('rotorRadius',60))
            offshore_p = OffshoreParams(waveHeight=i.get('waveHeight',1.8), wavePeriod=i.get('wavePeriod',6.5), currentSpeed=i.get('currentSpeed',0.9))
            battery_p  = BatteryParams(capacity=i.get('batteryCapacity',1500), currentCharge=i.get('batteryCurrentCharge',750), demand=i.get('demand',4000))
            cost_p     = CostParams()
            physics_res= calculate_all(solar_p, wind_p, offshore_p, battery_p, cost_p)
            surprise   = world_model.compute_surprise(physics_res, prediction)
        except Exception:
            pass

        return {
            'prediction': prediction,
            'confidence': world_model.metrics.get('confidence', None),
            'surprise':   surprise,
            '_meta':      {'elapsed_ms': round((time.time()-t0)*1000, 1), 'engine': 'world_model'},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/world-model/compare")
def wm_compare(req: CalculateRequest):
    """
    Run both physics engine AND world model, return side-by-side comparison.
    """
    t0 = time.time()
    try:
        solar_p, wind_p, offshore_p, battery_p, cost_p = dict_to_params(req)
        physics_res = calculate_all(solar_p, wind_p, offshore_p, battery_p, cost_p)

        wm_result = None
        surprise  = None
        if world_model.trained:
            inputs = {
                'panelTilt': solar_p.tilt, 'panelAzimuth': solar_p.azimuth,
                'panelArea': solar_p.panelArea, 'irradiance': solar_p.irradiance,
                'cellTemp': solar_p.cellTemp, 'dustLoss': solar_p.dustLoss,
                'shadeLoss': solar_p.shadeLoss, 'windSpeed': wind_p.windSpeed,
                'windDirection': wind_p.windDirection, 'yawAngle': wind_p.yawAngle,
                'rotorRadius': wind_p.rotorRadius, 'turbineSpacing': wind_p.turbineSpacing,
                'towerHeight': wind_p.hubHeight, 'waveHeight': offshore_p.waveHeight,
                'wavePeriod': offshore_p.wavePeriod, 'currentSpeed': offshore_p.currentSpeed,
                'batteryCapacity': battery_p.capacity,
                'batteryCurrentCharge': battery_p.currentCharge,
                'demand': battery_p.demand,
            }
            wm_result = world_model.predict(inputs)
            surprise  = world_model.compute_surprise(physics_res, wm_result)

        return {
            'physics':    physics_res,
            'world_model': wm_result,
            'surprise':   surprise,
            'confidence': world_model.metrics.get('confidence'),
            '_meta':      {'elapsed_ms': round((time.time()-t0)*1000, 1)},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/vr/sync")
def vr_sync(req: VRSyncRequest):
    """
    Unity/VR → Web Optimizer sync.
    Receives placed units + slider state from VR scene.
    Returns optimized placement suggestion.
    """
    n_solar    = sum(1 for u in req.units if u.get('type') == 'solar')
    n_wind     = sum(1 for u in req.units if u.get('type') == 'wind')
    n_offshore = sum(1 for u in req.units if u.get('type') == 'offshore')

    sl = req.sliders
    solar_p    = SolarParams(panelArea=n_solar*200 if n_solar>0 else 5000, tilt=sl.get('tilt',32), azimuth=sl.get('azimuth',180), irradiance=sl.get('irradiance',850))
    wind_p     = WindParams(turbineCount=max(1,n_wind), windSpeed=sl.get('windSpeed',8.5))
    offshore_p = OffshoreParams(waveHeight=sl.get('waveHeight',1.8))
    battery_p  = BatteryParams()
    cost_p     = CostParams()

    result = calculate_all(solar_p, wind_p, offshore_p, battery_p, cost_p)

    return {
        'scene_id':    req.scene_id,
        'unit_counts': {'solar': n_solar, 'wind': n_wind, 'offshore': n_offshore},
        'calculation': result,
        'suggestions': {
            'message': 'Calculation complete — results sent to VR dashboard',
            'optimal_tilt':    32,
            'optimal_azimuth': 180,
            'spacing_ok':      result['wind']['spacingOk'],
        },
        '_api_version': '2.0.0',
    }


# ── ROOT ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "RE Optimization Lab API",
        "version": "2.0.0",
        "docs":    "/docs",
        "health":  "/api/health",
        "by":      "ZSK Solutions",
    }
