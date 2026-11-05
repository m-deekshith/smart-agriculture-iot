"""
Master AI Inference Engine
Combines:
1. Multimodal Sensor Fusion Classifier (Soil + Optical Crop)
2. Crop Suitability Matcher (FAO / ICAR Agricultural Standards)
3. Weather-Gated Fertilizer Timing Advisory (Open-Meteo Integration)
"""

import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained models
model_soil   = joblib.load(f"{BASE_DIR}/model_soil.joblib")
model_crop   = joblib.load(f"{BASE_DIR}/model_crop.joblib")
model_concat = joblib.load(f"{BASE_DIR}/model_concat.joblib")

CLASS_NAMES = {
    0: "HEALTHY (Optimal Conditions)",
    1: "WATER_STRESS (Severe Drought Detected)",
    2: "NUTRIENT_STRESS (Chemical Deficit Detected)",
    3: "FOLIAR_STRESS (Heat Scorch / Leaf Disease)"
}

def predict_multimodal_health(soil_vector, crop_vector):
    """
    soil_vector: [moisture, temp, ec, ph, n, p, k]
    crop_vector: [canopy, exg, yellow]
    """
    s_feat = np.array([soil_vector])
    c_feat = np.array([crop_vector])

    # Probability vectors from both expert models
    prob_s = model_soil.predict_proba(s_feat)[0]
    prob_c = model_crop.predict_proba(c_feat)[0]

    # Weighted decision fusion (0.55 soil + 0.45 crop)
    prob_fused = (0.55 * prob_s) + (0.45 * prob_c)
    pred_class = int(np.argmax(prob_fused))
    confidence = float(np.max(prob_fused)) * 100.0

    return {
        "class_id": pred_class,
        "diagnosis": CLASS_NAMES[pred_class],
        "confidence_pct": round(confidence, 1),
        "soil_prob": [round(float(p), 3) for p in prob_s],
        "crop_prob": [round(float(p), 3) for p in prob_c]
    }

# Agronomic Crop Priority Table for stable deterministic sorting
CROP_PRIORITY = {
    "Wheat": 6,
    "Chickpea (Pulses)": 5,
    "Cotton": 4,
    "Maize (Corn)": 3,
    "Tomato": 2,
    "Rice (Paddy)": 1
}

# In-memory smoothed crop scores to prevent 2-second jittering
_smoothed_crop_scores = {}

def rank_crop_suitability(n, p, k, ph, moisture):
    """
    Ranks viable crops based on continuous agronomic envelope proximity.
    Uses continuous roll-off scoring and exponential smoothing (EMA) to prevent
    the top crops from jittering or flipping positions every 2 seconds due to
    micro-fluctuations in sensor ADC readings.
    """
    crops = [
        {"name": "Wheat", "min_n": 35, "max_n": 75, "min_p": 25, "max_p": 55, "min_k": 30, "max_k": 65, "min_ph": 6.0, "max_ph": 7.5, "min_m": 35, "max_m": 60},
        {"name": "Chickpea (Pulses)", "min_n": 10, "max_n": 40, "min_p": 35, "max_p": 65, "min_k": 45, "max_k": 85, "min_ph": 6.2, "max_ph": 8.0, "min_m": 25, "max_m": 45},
        {"name": "Maize (Corn)", "min_n": 50, "max_n": 85, "min_p": 30, "max_p": 60, "min_k": 35, "max_k": 70, "min_ph": 5.8, "max_ph": 7.2, "min_m": 40, "max_m": 65},
        {"name": "Cotton", "min_n": 60, "max_n": 100, "min_p": 25, "max_p": 50, "min_k": 55, "max_k": 95, "min_ph": 6.5, "max_ph": 8.2, "min_m": 35, "max_m": 55},
        {"name": "Rice (Paddy)", "min_n": 55, "max_n": 95, "min_p": 30, "max_p": 60, "min_k": 40, "max_k": 70, "min_ph": 5.5, "max_ph": 7.0, "min_m": 65, "max_m": 90},
        {"name": "Tomato", "min_n": 45, "max_n": 80, "min_p": 45, "max_p": 75, "min_k": 60, "max_k": 100, "min_ph": 6.0, "max_ph": 6.8, "min_m": 45, "max_m": 65}
    ]

    def param_score(val, min_v, max_v):
        if min_v <= val <= max_v:
            return 1.0
        width = max(max_v - min_v, 1.0)
        dist = (min_v - val) if val < min_v else (val - max_v)
        return max(0.0, 1.0 - (dist / (width * 1.5)))

    rankings = []
    for c in crops:
        s_n  = param_score(n, c["min_n"], c["max_n"])
        s_p  = param_score(p, c["min_p"], c["max_p"])
        s_k  = param_score(k, c["min_k"], c["max_k"])
        s_ph = param_score(ph, c["min_ph"], c["max_ph"])
        s_m  = param_score(moisture, c["min_m"], c["max_m"])

        raw_pct = ((s_n * 0.20) + (s_p * 0.20) + (s_k * 0.20) + (s_ph * 0.20) + (s_m * 0.20)) * 100.0

        # Temporal EMA smoothing: smooth 85% old + 15% new
        prev = _smoothed_crop_scores.get(c["name"], raw_pct)
        smoothed = (0.85 * prev) + (0.15 * raw_pct)
        _smoothed_crop_scores[c["name"]] = smoothed

        rankings.append({
            "crop": c["name"],
            "match_pct": round(smoothed, 1),
            "priority": CROP_PRIORITY.get(c["name"], 0)
        })

    # Stable deterministic sort: primary by match_pct, secondary by crop priority
    rankings.sort(key=lambda x: (round(x["match_pct"], 1), x["priority"]), reverse=True)
    return rankings

def get_fertilizer_and_timing_advisory(n, p, k, moisture, rain_prob_24h):
    """
    Calculates exact fertilizer dosages and verifies weather-gated timing
    """
    TARGET_N, TARGET_P, TARGET_K = 50.0, 40.0, 50.0
    fertilizers = []

    if n < TARGET_N:
        urea_dose = round(((TARGET_N - n) * 2.24) / 0.46, 1)
        fertilizers.append(f"Urea (46% N) @ {urea_dose} kg/ha")
    if p < TARGET_P:
        dap_dose = round(((TARGET_P - p) * 2.24) / 0.46, 1)
        fertilizers.append(f"DAP (18% N, 46% P2O5) @ {dap_dose} kg/ha")
    if k < TARGET_K:
        mop_dose = round(((TARGET_K - k) * 2.24) / 0.60, 1)
        fertilizers.append(f"MOP (60% K2O) @ {mop_dose} kg/ha")

    # Timing Decision Logic
    if not fertilizers:
        timing_status = "OPTIMAL"
        timing_message = "✅ Soil nutrients are optimal. No chemical fertilizer required."
    elif rain_prob_24h > 50.0 or moisture > 65.0:
        timing_status = "BLOCKED_LEACHING_RISK"
        timing_message = (
            f"🛑 DEFER APPLICATION: Rain probability ({rain_prob_24h}%) or soil moisture ({moisture}%) is too high. "
            "Fertilizer will be washed into groundwater!"
        )
    elif moisture < 20.0:
        timing_status = "BLOCKED_DRY_SOIL"
        timing_message = (
            f"⚠️ SOIL TOO DRY ({moisture}%): Fertilizer granules cannot dissolve and will scorch roots. "
            "Irrigate first, then apply."
        )
    else:
        timing_status = "APPROVED_APPLICATION"
        timing_message = (
            f"✅ APPROVED FOR IMMEDIATE APPLICATION: Soil moisture ({moisture}%) is ideal for root uptake. "
            "Apply in early morning or evening."
        )

    return {
        "fertilizers": fertilizers,
        "timing_status": timing_status,
        "timing_message": timing_message
    }

def run_master_ai_analysis(soil_dict, crop_dict, weather_dict):
    """
    Master unified entry-point for the Web Dashboard & Gateway
    """
    soil_vec = [
        soil_dict["moisture"], soil_dict["temp"], soil_dict["ec"],
        soil_dict["ph"], soil_dict["n"], soil_dict["p"], soil_dict["k"]
    ]
    crop_vec = [
        crop_dict["canopy_cover_pct"], crop_dict["exg"], crop_dict["yellowing_pct"]
    ]

    health_diag = predict_multimodal_health(soil_vec, crop_vec)
    crop_ranks  = rank_crop_suitability(soil_dict["n"], soil_dict["p"], soil_dict["k"], soil_dict["ph"], soil_dict["moisture"])
    fert_advice = get_fertilizer_and_timing_advisory(
        soil_dict["n"], soil_dict["p"], soil_dict["k"],
        soil_dict["moisture"], weather_dict.get("rain_prob_24h", 10.0)
    )

    # Weather-gated Irrigation decision
    irrigation_action = "PUMP_OFF"
    irrigation_reason = "Soil moisture is in optimal vegetative range."
    if health_diag["class_id"] == 1 or soil_dict["moisture"] < 22.0:
        if weather_dict.get("rain_prob_next_4h", 0) > 60.0:
            irrigation_action = "PUMP_DEFERRED"
            irrigation_reason = f"Soil is dry, BUT rain expected ({weather_dict.get('rain_prob_next_4h')}%). Irrigation held to save water."
        else:
            irrigation_action = "PUMP_ON"
            irrigation_reason = "Severe soil drought detected with no imminent rain. Drip irrigation active."
    elif soil_dict["moisture"] > 75.0:
        irrigation_action = "EMERGENCY_CUTOFF"
        irrigation_reason = "Waterlogging hazard detected! All irrigation lines locked."

    return {
        "health_diagnosis": health_diag,
        "crop_rankings": crop_ranks[:3], # Top 3
        "fertilizer_advisory": fert_advice,
        "irrigation_decision": {
            "action": irrigation_action,
            "reason": irrigation_reason
        }
    }

if __name__ == "__main__":
    # Test sample
    s = {"moisture": 19.5, "temp": 28.4, "ec": 410, "ph": 6.8, "n": 22, "p": 18, "k": 62}
    c = {"canopy_cover_pct": 52.0, "exg": 0.12, "yellowing_pct": 8.5}
    w = {"rain_prob_next_4h": 75.0, "rain_prob_24h": 80.0}

    res = run_master_ai_analysis(s, c, w)
    print("\n" + "="*60)
    print("🌾 MASTER AI ANALYSIS RESULT TEST")
    print("="*60)
    print("1. Diagnosis:", res["health_diagnosis"]["diagnosis"], f"({res['health_diagnosis']['confidence_pct']}%)")
    print("2. Irrigation Action:", res["irrigation_decision"]["action"], "->", res["irrigation_decision"]["reason"])
    print("3. Best Crop:", res["crop_rankings"][0]["crop"], f"({res['crop_rankings'][0]['match_pct']}%)")
    print("4. Fertilizer Timing:", res["fertilizer_advisory"]["timing_message"])
