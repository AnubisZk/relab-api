"""
================================================================
WORLD MODEL — Surrogate ML (scikit-learn MLPRegressor)
RE Optimization Lab · ZSK Solutions
----------------------------------------------------------------
JEPA-inspired surrogate layer:
- Trained on physics engine simulation history
- Provides fast prediction (<5ms vs ~50ms grid search)
- Does NOT replace physics engine for final validation
- Computes surprise/anomaly score vs physics result
- confidence = 1 - mean_abs_error on validation set
================================================================
"""

import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

from physics_engine import (
    SolarParams, WindParams, OffshoreParams, BatteryParams, CostParams,
    calculate_all
)

MODEL_PATH  = Path("world_model.joblib")
SCALER_PATH = Path("world_model_scaler.joblib")

# Feature vector order — must match JS worldModel.js
FEATURE_NAMES = [
    'panelTilt', 'panelAzimuth', 'panelArea', 'irradiance', 'cellTemp',
    'dustLoss', 'shadeLoss', 'windSpeed', 'windDirection', 'yawAngle',
    'rotorRadius', 'turbineSpacing', 'towerHeight',
    'waveHeight', 'wavePeriod', 'currentSpeed',
    'batteryCapacity', 'batteryCurrentCharge', 'demand',
]

TARGET_NAMES = [
    'E_solar_net', 'E_wind_net', 'E_offshore_net',
    'battery_final', 'total_cost', 'total_loss', 'Z',
    'risk_score', 'constraint_violation_prob',
]


class WorldModel:
    def __init__(self):
        self.model         = None
        self.scaler        = None
        self.target_scaler = None
        self.trained       = False
        self.metrics       = {}
        self._try_load()

    def _try_load(self):
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            try:
                self.model   = joblib.load(MODEL_PATH)
                self.scaler  = joblib.load(SCALER_PATH)
                ts_path = Path("world_model_target_scaler.joblib")
                self.target_scaler = joblib.load(ts_path) if ts_path.exists() else None
                self.trained = True
                print("[WorldModel] Loaded saved model")
            except Exception as e:
                print(f"[WorldModel] Load failed: {e}")

    # ── FEATURE EXTRACTION ─────────────────────────────────────

    def extract_features(self, inputs: dict) -> np.ndarray:
        """Convert API input dict → feature vector"""
        return np.array([[
            inputs.get('panelTilt', 32),
            inputs.get('panelAzimuth', 180),
            inputs.get('panelArea', 5000),
            inputs.get('irradiance', 850),
            inputs.get('cellTemp', 35),
            inputs.get('dustLoss', 0.15),
            inputs.get('shadeLoss', 0.05),
            inputs.get('windSpeed', 8.5),
            inputs.get('windDirection', 230),
            inputs.get('yawAngle', 0),
            inputs.get('rotorRadius', 60),
            inputs.get('turbineSpacing', 7),
            inputs.get('towerHeight', 100),
            inputs.get('waveHeight', 1.8),
            inputs.get('wavePeriod', 6.5),
            inputs.get('currentSpeed', 0.9),
            inputs.get('batteryCapacity', 1500),
            inputs.get('batteryCurrentCharge', 750),
            inputs.get('demand', 4000),
        ]])

    # ── DATA GENERATION ────────────────────────────────────────

    def generate_training_data(self, n_samples: int = 5000) -> tuple:
        """
        Generate n_samples by running physics engine with random params.
        This is the 'simulation history' the surrogate learns from.
        """
        rng = np.random.default_rng(42)
        X, y = [], []

        for _ in range(n_samples):
            # Random params within realistic ranges
            tilt     = rng.uniform(0, 60)
            azimuth  = rng.uniform(90, 270)
            area     = rng.uniform(100, 10000)
            irr      = rng.uniform(200, 1200)
            cellTemp = rng.uniform(15, 75)
            dust     = rng.uniform(0, 0.5)
            shade    = rng.uniform(0, 0.4)
            wspd     = rng.uniform(2, 20)
            wdir     = rng.uniform(0, 360)
            yaw      = rng.uniform(0, 30)
            rotorR   = rng.uniform(20, 80)
            spacing  = rng.uniform(3, 12)
            tower    = rng.uniform(60, 150)
            waveH    = rng.uniform(0.1, 8)
            waveP    = rng.uniform(3, 18)
            curr     = rng.uniform(0, 2.5)
            bcap     = rng.uniform(50, 3000)
            bcur     = rng.uniform(0, bcap)
            demand   = rng.uniform(500, 8000)

            solar_p = SolarParams(
                panelArea=area, irradiance=irr, efficiency=0.225,
                timeHours=8, tilt=tilt, azimuth=azimuth,
                cellTemp=cellTemp, refTemp=25, tempCoeff=0.0045,
                dustLoss=dust, shadeLoss=shade, cleaningCost=120, cleaningInterval=14
            )
            wind_p = WindParams(
                turbineCount=5, rotorRadius=rotorR, hubHeight=tower,
                windSpeed=wspd, windDirection=wdir, yawAngle=yaw,
                powerCoeff=0.45, airDensity=1.225, turbineSpacing=spacing,
                wakeLoss=0.05, turbulenceLoss=0.05
            )
            offshore_p = OffshoreParams(
                waveHeight=waveH, wavePeriod=waveP, currentSpeed=curr,
                platformMotion=0.6, corrosionLoss=0.08, waveLoss=0.10,
                maintenanceCost=35000, safetyThreshold=100
            )
            battery_p = BatteryParams(
                capacity=bcap, currentCharge=bcur,
                chargeEfficiency=0.95, dischargeEfficiency=0.95,
                demand=demand, minimumReserve=bcap*0.2
            )
            cost_p = CostParams()

            try:
                res = calculate_all(solar_p, wind_p, offshore_p, battery_p, cost_p)
                violations = [v for v in res['violations'] if v['severity']=='error']
                X.append([tilt, azimuth, area, irr, cellTemp, dust, shade,
                          wspd, wdir, yaw, rotorR, spacing, tower,
                          waveH, waveP, curr, bcap, bcur, demand])
                y.append([
                    res['solar']['E_net'],
                    res['wind']['E_net'],
                    res['offshore']['E_net'],
                    res['battery']['B_next'],
                    res['objective']['total_cost'],
                    res['objective']['total_loss'],
                    res['objective']['Z'],
                    res['objective']['risk_score'],
                    float(len(violations) > 0),
                ])
            except Exception:
                continue

        return np.array(X), np.array(y)

    # ── TRAINING ───────────────────────────────────────────────

    def train(self, n_samples: int = 5000) -> dict:
        """
        Train MLPRegressor on physics engine simulation data.
        Uses log1p transform on targets to handle large value ranges.
        Architecture: 3 hidden layers [128, 64, 32] with relu activation.
        """
        print(f"[WorldModel] Generating {n_samples} training samples...")
        X, y = self.generate_training_data(n_samples)

        # Log1p transform targets to normalize large ranges
        # (E values can range 0–100,000+ kWh)
        y_log = np.sign(y) * np.log1p(np.abs(y))

        X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.15, random_state=42)

        self.scaler = StandardScaler()
        X_train_s   = self.scaler.fit_transform(X_train)
        X_val_s     = self.scaler.transform(X_val)

        # Also scale targets
        self.target_scaler = StandardScaler()
        y_train_s = self.target_scaler.fit_transform(y_train)

        print("[WorldModel] Training MLPRegressor...")
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        )
        self.model.fit(X_train_s, y_train_s)

        # Validation — inverse transform back to original scale
        y_pred_s   = self.model.predict(X_val_s)
        y_pred_log = self.target_scaler.inverse_transform(y_pred_s)
        y_pred     = np.sign(y_pred_log) * (np.expm1(np.abs(y_pred_log)))
        y_val_orig = np.sign(y_val) * (np.expm1(np.abs(y_val)))

        mape_per_target = []
        for i, name in enumerate(TARGET_NAMES):
            try:
                mask = np.abs(y_val_orig[:, i]) > 1.0  # skip near-zero
                if mask.sum() < 10:
                    mape_per_target.append(0.0)
                    continue
                mape = mean_absolute_percentage_error(
                    y_val_orig[mask, i], y_pred[mask, i]) * 100
                mape = min(mape, 100.0)  # cap at 100%
            except Exception:
                mape = 0.0
            mape_per_target.append(round(mape, 2))

        mean_mape  = np.mean(mape_per_target)
        confidence = max(0.0, min(100.0, 100 - mean_mape))

        self.metrics = {
            'n_samples':  len(X),
            'n_train':    len(X_train),
            'n_val':      len(X_val),
            'mean_mape':  round(float(mean_mape), 2),
            'confidence': round(float(confidence), 1),
            'per_target': dict(zip(TARGET_NAMES, mape_per_target)),
        }
        self.trained = True

        joblib.dump(self.model,         MODEL_PATH)
        joblib.dump(self.scaler,        SCALER_PATH)
        joblib.dump(self.target_scaler, Path("world_model_target_scaler.joblib"))
        print(f"[WorldModel] Done. Confidence: {confidence:.1f}%")
        return self.metrics

    # ── PREDICTION ─────────────────────────────────────────────

    def predict(self, inputs: dict) -> dict:
        """
        Fast surrogate prediction. Inverse transforms log-scaled targets.
        """
        if not self.trained:
            raise RuntimeError("World model not trained.")

        X   = self.extract_features(inputs)
        X_s = self.scaler.transform(X)
        y_pred_s = self.model.predict(X_s)

        # Inverse target scaling + inverse log1p
        if self.target_scaler is not None:
            y_pred_log = self.target_scaler.inverse_transform(y_pred_s)
        else:
            y_pred_log = y_pred_s
        y_pred = np.sign(y_pred_log) * (np.expm1(np.abs(y_pred_log)))

        return {name: float(val) for name, val in zip(TARGET_NAMES, y_pred[0])}

    # ── SURPRISE SCORE ─────────────────────────────────────────

    def compute_surprise(self, physics_result: dict, wm_prediction: dict) -> dict:
        """
        surprise = |physics - wm_pred| / max(|physics|, ε)
        If surprise > 0.15 → anomaly flag
        """
        EPSILON = 1e-9
        THRESHOLD = 0.15

        pairs = {
            'E_solar_net':  (physics_result.get('solar',{}).get('E_net', 0),    wm_prediction.get('E_solar_net', 0)),
            'E_wind_net':   (physics_result.get('wind',{}).get('E_net', 0),     wm_prediction.get('E_wind_net', 0)),
            'E_offshore_net':(physics_result.get('offshore',{}).get('E_net',0), wm_prediction.get('E_offshore_net', 0)),
            'Z':            (physics_result.get('objective',{}).get('Z', 0),    wm_prediction.get('Z', 0)),
        }

        surprises = {}
        for key, (phys, wm) in pairs.items():
            surprises[key] = abs(phys - wm) / max(abs(phys), EPSILON)

        mean_surprise  = float(np.mean(list(surprises.values())))
        anomaly_detected = mean_surprise > THRESHOLD

        return {
            'per_metric':       surprises,
            'mean_surprise':    round(mean_surprise, 4),
            'anomaly_detected': anomaly_detected,
            'threshold':        THRESHOLD,
            'message': 'Anomalous simulation state detected — verify inputs' if anomaly_detected else 'Normal',
        }


# Singleton
world_model = WorldModel()
