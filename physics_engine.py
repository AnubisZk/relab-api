"""
================================================================
PHYSICS ENGINE — Python port
RE Optimization Lab · ZSK Solutions
----------------------------------------------------------------
All formulas match physicsEngine.js exactly.
Every result derived from physical equations — no fake scores.
================================================================
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

DEG2RAD = math.pi / 180
EPSILON  = 1e-9
ENERGY_PRICE = 0.12  # $/kWh


# ── INPUT MODELS ────────────────────────────────────────────

@dataclass
class SolarParams:
    panelArea:        float = 5000.0   # m²
    irradiance:       float = 850.0    # W/m²
    efficiency:       float = 0.225    # fraction
    timeHours:        float = 8.0      # h
    tilt:             float = 32.0     # degrees
    azimuth:          float = 180.0    # degrees
    cellTemp:         float = 35.0     # °C
    refTemp:          float = 25.0     # °C
    tempCoeff:        float = 0.0045   # fraction/°C
    dustLoss:         float = 0.15     # fraction
    shadeLoss:        float = 0.05     # fraction
    cleaningCost:     float = 120.0    # $
    cleaningInterval: float = 14.0     # days

@dataclass
class WindParams:
    turbineCount:   int   = 5
    rotorRadius:    float = 60.0    # m
    hubHeight:      float = 100.0   # m
    windSpeed:      float = 8.5     # m/s
    windDirection:  float = 230.0   # degrees
    yawAngle:       float = 0.0     # degrees misalignment
    powerCoeff:     float = 0.45    # Cp
    airDensity:     float = 1.225   # kg/m³
    turbineSpacing: float = 7.0     # × rotor diameter
    wakeLoss:       float = 0.12    # fraction
    turbulenceLoss: float = 0.05    # fraction

@dataclass
class OffshoreParams:
    waveHeight:      float = 1.8    # m Hs
    wavePeriod:      float = 6.5    # s Tp
    currentSpeed:    float = 0.9    # m/s
    platformMotion:  float = 0.6    # degrees
    corrosionLoss:   float = 0.08   # fraction
    waveLoss:        float = 0.10   # fraction
    maintenanceCost: float = 35000  # $/yr
    safetyThreshold: float = 100.0

@dataclass
class BatteryParams:
    capacity:            float = 1500.0  # kWh
    currentCharge:       float = 750.0   # kWh
    chargeEfficiency:    float = 0.95
    dischargeEfficiency: float = 0.95
    demand:              float = 4000.0  # kWh/day
    minimumReserve:      float = 300.0   # kWh

@dataclass
class CostParams:
    installationBudget: float = 2500000.0
    availableArea:      float = 20000.0
    co2Factor:          float = 0.45  # kg CO₂/kWh


# ── SOLAR MODEL ─────────────────────────────────────────────

def calc_solar(p: SolarParams) -> dict:
    """
    E_solar = A × (G/1000) × η × t
    L_temp  = β × (T_cell - T_ref)
    F_angle = max(0, cos(|tilt - 30|))
    F_az    = max(0.5, cos(|azimuth - 180|))
    C_f     = 1 - dustLoss
    E_net   = E_gross × F_angle × F_az × C_f × (1-shade) × max(0,1-L_temp)
    """
    E_gross = p.panelArea * (p.irradiance / 1000) * p.efficiency * p.timeHours
    L_temp  = max(0.0, p.tempCoeff * (p.cellTemp - p.refTemp))
    F_angle = max(0.0, math.cos(abs(p.tilt - 30) * DEG2RAD))
    F_az    = max(0.5, math.cos(abs(p.azimuth - 180) * DEG2RAD))
    C_f     = 1.0 - p.dustLoss

    # Cleaning decision: is energy recovery worth the cost?
    energy_recovered_val = p.dustLoss * E_gross * ENERGY_PRICE
    clean_worth = energy_recovered_val > p.cleaningCost
    C_f_eff     = 1.0 if clean_worth else C_f
    cleaning_cost_applied = p.cleaningCost if clean_worth else 0.0

    E_net = E_gross * F_angle * F_az * C_f_eff \
          * (1 - p.shadeLoss) * max(0.0, 1 - L_temp)

    loss_temp  = E_gross * L_temp
    loss_dust  = E_gross * p.dustLoss
    loss_shade = E_gross * p.shadeLoss
    loss_angle = E_gross * (1 - F_angle)

    P_peak = p.panelArea * p.efficiency  # kWp
    E_ref  = p.panelArea * (p.irradiance / 1000) * p.timeHours
    PR     = min(1.0, E_net / max(EPSILON, E_ref))
    SY     = E_net / max(EPSILON, P_peak)

    return dict(
        E_gross=E_gross, E_net=max(0.0, E_net),
        L_temp=L_temp, F_angle=F_angle, F_az=F_az, C_f=C_f_eff,
        loss_temp=max(0.0, loss_temp), loss_dust=max(0.0, loss_dust),
        loss_shade=max(0.0, loss_shade), loss_angle=max(0.0, loss_angle),
        PR=PR, SY=SY, P_peak=P_peak,
        clean_worth=bool(clean_worth),
        cleaning_cost_applied=float(cleaning_cost_applied),
        energy_recovered_val=float(energy_recovered_val),
        formula=f"E={p.panelArea}×{p.irradiance/1000:.3f}×{p.efficiency}×{p.timeHours}"
                f"×cos({abs(p.tilt-30):.1f}°)×{C_f_eff:.3f}×(1-{p.shadeLoss})×(1-{L_temp:.4f})"
    )


# ── WIND MODEL ──────────────────────────────────────────────

def calc_wake_loss(Cp: float, rotorD: float, spacing_mult: float, k: float = 0.04) -> float:
    """
    Jensen wake model:
    deficit = (1 - sqrt(1-Cp)) × (D / (D + 2·k·x))²
    x = spacing_mult × D
    """
    x = spacing_mult * rotorD
    deficit = (1 - math.sqrt(max(0.0, 1 - Cp))) \
            * (rotorD / (rotorD + 2 * k * x)) ** 2
    return min(0.5, max(0.0, deficit))


def calc_wind(p: WindParams, time_hours: float) -> dict:
    """
    A_rotor = π × r²
    P_wind  = 0.5 × ρ × A_rotor × v³ × Cp   [kW]
    P_yaw   = P_wind × cos³(γ)
    P_net   = P_yaw × (1-L_wake) × (1-L_turb) × N
    E_wind  = P_net × t                        [kWh]
    """
    # Cut-in / cut-out
    if p.windSpeed < 3.0 or p.windSpeed > 25.0:
        return dict(E_net=0.0, P_wind_single=0.0, P_yaw=0.0, P_net=0.0,
                    L_wake_eff=0.0, loss_wake=0.0, loss_turbulence=0.0, loss_yaw=0.0,
                    spacingOk=p.turbineSpacing >= 7, rotorD=p.rotorRadius*2,
                    formula=f"v={p.windSpeed} outside cut-in/cut-out → P=0")

    A_rotor   = math.pi * p.rotorRadius ** 2
    P_single  = 0.5 * p.airDensity * A_rotor * p.windSpeed**3 * p.powerCoeff / 1000  # kW
    gamma     = p.yawAngle * DEG2RAD
    P_yaw_val = P_single * math.cos(gamma) ** 3

    rotorD          = p.rotorRadius * 2
    L_wake_jensen   = calc_wake_loss(p.powerCoeff, rotorD, p.turbineSpacing)
    L_wake_eff      = max(L_wake_jensen, p.wakeLoss)

    P_net_1T = P_yaw_val * (1 - L_wake_eff) * (1 - p.turbulenceLoss)
    P_farm   = P_net_1T * p.turbineCount  # kW
    E_net    = P_farm * time_hours         # kWh

    loss_wake       = P_yaw_val * L_wake_eff * p.turbineCount * time_hours
    loss_turbulence = P_yaw_val * (1-L_wake_eff) * p.turbulenceLoss * p.turbineCount * time_hours
    loss_yaw        = (P_single - P_yaw_val) * p.turbineCount * time_hours

    return dict(
        E_net=max(0.0, E_net), P_wind_single=P_single, P_yaw=P_yaw_val,
        P_net=P_farm, L_wake_eff=L_wake_eff,
        loss_wake=max(0.0, loss_wake),
        loss_turbulence=max(0.0, loss_turbulence),
        loss_yaw=max(0.0, loss_yaw),
        spacingOk=bool(p.turbineSpacing >= 7), rotorD=rotorD,
        formula=f"P=½×{p.airDensity}×π×{p.rotorRadius}²×{p.windSpeed}³×{p.powerCoeff}"
                f"={P_single:.2f}kW×cos³({p.yawAngle}°)×(1-{L_wake_eff:.3f})×{p.turbineCount}T"
    )


# ── OFFSHORE MODEL ──────────────────────────────────────────

def calc_offshore(p: OffshoreParams, E_wind_kWh: float) -> dict:
    """
    R_sea = w1·Hs + w2·Tp + w3·v_cur + w4·δ_osc
    Weights: 0.40, 0.15, 0.25, 0.20
    E_offshore = E_wind × (1-L_wave) × (1-L_corr) - maintenance_daily_kWh
    """
    w1, w2, w3, w4 = 0.40, 0.15, 0.25, 0.20
    R_sea = (w1 * p.waveHeight + w2 * p.wavePeriod
           + w3 * p.currentSpeed + w4 * p.platformMotion)

    maint_daily_kWh = (p.maintenanceCost / 365) / ENERGY_PRICE
    E_net_raw = E_wind_kWh * (1 - p.waveLoss) * (1 - p.corrosionLoss) - maint_daily_kWh
    E_net     = max(0.0, E_net_raw)

    loss_wave      = E_wind_kWh * p.waveLoss
    loss_corrosion = E_wind_kWh * p.corrosionLoss

    return dict(
        R_sea=R_sea, E_net=E_net, maint_kWh=maint_daily_kWh,
        loss_wave=max(0.0, loss_wave),
        loss_corrosion=max(0.0, loss_corrosion),
        risk_violation=bool(R_sea > p.safetyThreshold),
        maintenance_accessible=bool(p.waveHeight < 3.5),
        formula=f"R={w1}×{p.waveHeight}+{w2}×{p.wavePeriod}+{w3}×{p.currentSpeed}"
                f"+{w4}×{p.platformMotion}={R_sea:.3f}"
    )


# ── BATTERY MODEL ────────────────────────────────────────────

def calc_battery(p: BatteryParams, E_available: float) -> dict:
    """
    B_next = B_current + η_c × E_in - E_out / η_d
    Constraint: 0 ≤ B_next ≤ B_capacity
    """
    E_in      = min(p.capacity - p.currentCharge, max(0.0, E_available))
    B_charged = p.currentCharge + p.chargeEfficiency * E_in
    B_dispatch = max(0.0, B_charged - p.minimumReserve)
    E_out     = min(p.demand, B_dispatch)
    B_next    = max(0.0, min(p.capacity, B_charged - E_out / p.dischargeEfficiency))
    unmet     = max(0.0, p.demand - E_out)
    B_avail   = max(0.0, B_next - p.minimumReserve)
    SOC       = (B_next / p.capacity) * 100 if p.capacity > 0 else 0

    return dict(
        B_next=B_next, B_available=B_avail, E_in=E_in, E_out=E_out,
        unmet=unmet, SOC=SOC,
        reserve_violation=bool(B_next < p.minimumReserve),
        formula=f"B_next={p.currentCharge}+{p.chargeEfficiency}×{E_in:.1f}"
                f"-{E_out:.1f}/{p.dischargeEfficiency}={B_next:.2f} kWh"
    )


# ── OBJECTIVE FUNCTION ───────────────────────────────────────

def calc_objective(s_res, w_res, o_res, b_res, solar_p, offshore_p, cost_p) -> dict:
    """
    Z = E_solar_net + E_wind_net + E_offshore_net + B_available
        - C_cleaning/0.12 - C_maintenance_daily/0.12 - C_install_daily/0.12 - C_risk/0.12
    """
    RISK_PENALTY = 10.0
    total_energy = s_res['E_net'] + w_res['E_net'] + o_res['E_net']
    C_cleaning   = s_res['cleaning_cost_applied']
    C_maint      = offshore_p.maintenanceCost / 365
    C_install    = cost_p.installationBudget / 365 / 20
    C_risk       = o_res['R_sea'] * RISK_PENALTY
    total_cost   = C_cleaning + C_maint + C_install + C_risk

    Z = total_energy + b_res['B_available'] - total_cost / ENERGY_PRICE

    total_loss = (s_res.get('loss_temp',0) + s_res.get('loss_dust',0) +
                  s_res.get('loss_shade',0) + w_res.get('loss_wake',0) +
                  w_res.get('loss_turbulence',0) + o_res.get('loss_wave',0) +
                  o_res.get('loss_corrosion',0))

    coverage  = min(100.0, total_energy / max(EPSILON, b_res.get('unmet',1)+total_energy) * 100) if total_energy > 0 else 0
    # simpler: coverage = total_energy / demand * 100
    co2_saved = total_energy * cost_p.co2Factor
    risk_score= min(100.0, o_res['R_sea'] * 10)

    return dict(
        Z=Z, total_energy=total_energy, total_cost=total_cost,
        total_loss=total_loss, coverage=coverage,
        co2_saved=co2_saved, risk_score=risk_score,
        C_cleaning=C_cleaning, C_maint=C_maint,
        C_install=C_install, C_risk=C_risk,
    )


# ── CONSTRAINT CHECKER ───────────────────────────────────────

def check_constraints(solar_p, wind_p, o_res, b_res, w_res, cost_p) -> list:
    violations = []
    if float(solar_p.panelArea) > float(cost_p.availableArea):
        violations.append({'id':'area','msg':f"Panel area {solar_p.panelArea}m² > available {cost_p.availableArea}m²",'severity':'error'})
    if not bool(w_res.get('spacingOk', True)):
        violations.append({'id':'spacing','msg':f"Spacing {wind_p.turbineSpacing}D < required 7D",'severity':'error'})
    if bool(b_res.get('reserve_violation')):
        violations.append({'id':'reserve','msg':f"Battery final {b_res['B_next']:.1f} kWh < min reserve",'severity':'error'})
    if bool(o_res.get('risk_violation')):
        violations.append({'id':'risk','msg':f"R_sea={o_res['R_sea']:.2f} > safety threshold",'severity':'error'})
    if not (0 <= float(solar_p.tilt) <= 90):
        violations.append({'id':'tilt','msg':f"Tilt {solar_p.tilt}° outside [0°, 90°]",'severity':'error'})
    if not (0 <= float(wind_p.yawAngle) <= 180):
        violations.append({'id':'yaw','msg':f"Yaw {wind_p.yawAngle}° outside [0°, 180°]",'severity':'error'})
    if float(b_res.get('unmet', 0)) > 0:
        violations.append({'id':'unmet','msg':f"Unmet demand: {b_res['unmet']:.1f} kWh/day",'severity':'warning'})
    if not bool(o_res.get('maintenance_accessible', True)):
        violations.append({'id':'access','msg':f"Hs > 3.5m — maintenance access restricted",'severity':'warning'})
    return violations


# ── MASTER CALCULATE ─────────────────────────────────────────

def calculate_all(solar_p: SolarParams, wind_p: WindParams,
                  offshore_p: OffshoreParams, battery_p: BatteryParams,
                  cost_p: CostParams) -> dict:

    s_res = calc_solar(solar_p)
    # patch key name for objective
    s_res['cleaning_cost_applied'] = s_res['cleaning_cost_applied'] if 'cleaning_cost_applied' in s_res else s_res.get('cleaningCostApplied', 0)
    w_res = calc_wind(wind_p, solar_p.timeHours)
    o_res = calc_offshore(offshore_p, w_res['E_net'])
    total_avail = s_res['E_net'] + w_res['E_net'] + o_res['E_net']
    b_res = calc_battery(battery_p, total_avail)
    z_res = calc_objective(s_res, w_res, o_res, b_res, solar_p, offshore_p, cost_p)
    violations = check_constraints(solar_p, wind_p, o_res, b_res, w_res, cost_p)

    return dict(
        solar=s_res, wind=w_res, offshore=o_res,
        battery=b_res, objective=z_res, violations=violations
    )
