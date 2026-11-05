"""
Agricultural Crop Recommendation & Fertilizer Timing Advisory Engine
Based on 7-in-1 Soil Sensor Data (N, P, K, pH, Moisture, EC, Temp) + Weather Gating
"""

def evaluate_crop_suitability(n, p, k, ph, moisture, temp):
    """
    Ranks suitable crops based on agronomic bounds (FAO / ICAR standards)
    """
    crops = [
        {"name": "Wheat", "min_n": 35, "max_n": 75, "min_p": 25, "max_p": 55, "min_k": 30, "max_k": 65, "min_ph": 6.0, "max_ph": 7.5, "min_m": 35, "max_m": 60},
        {"name": "Rice (Paddy)", "min_n": 55, "max_n": 95, "min_p": 30, "max_p": 60, "min_k": 40, "max_k": 70, "min_ph": 5.5, "max_ph": 7.0, "min_m": 65, "max_m": 90},
        {"name": "Maize (Corn)", "min_n": 50, "max_n": 85, "min_p": 30, "max_p": 60, "min_k": 35, "max_k": 70, "min_ph": 5.8, "max_ph": 7.2, "min_m": 40, "max_m": 65},
        {"name": "Chickpea (Pulses)", "min_n": 10, "max_n": 40, "min_p": 35, "max_p": 65, "min_k": 45, "max_k": 85, "min_ph": 6.2, "max_ph": 8.0, "min_m": 25, "max_m": 45},
        {"name": "Cotton", "min_n": 60, "max_n": 100, "min_p": 25, "max_p": 50, "min_k": 55, "max_k": 95, "min_ph": 6.5, "max_ph": 8.2, "min_m": 35, "max_m": 55},
        {"name": "Tomato", "min_n": 45, "max_n": 80, "min_p": 45, "max_p": 75, "min_k": 60, "max_k": 100, "min_ph": 6.0, "max_ph": 6.8, "min_m": 45, "max_m": 65}
    ]

    rankings = []
    for c in crops:
        score = 0
        total_checks = 5
        if c["min_n"] <= n <= c["max_n"]: score += 1
        if c["min_p"] <= p <= c["max_p"]: score += 1
        if c["min_k"] <= k <= c["max_k"]: score += 1
        if c["min_ph"] <= ph <= c["max_ph"]: score += 1
        if c["min_m"] <= moisture <= c["max_m"]: score += 1
        match_pct = round((score / total_checks) * 100.0, 1)
        rankings.append((c["name"], match_pct))

    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings

def calculate_fertilizer_advisory(n, p, k, moisture, rain_prob_next_24h):
    """
    Calculates nutrient deficit and determines application timing safety
    Target baseline for general cereal/crop: N=50, P=40, K=50 mg/kg
    """
    TARGET_N, TARGET_P, TARGET_K = 50.0, 40.0, 50.0
    deficits = {}
    recommendations = []

    if n < TARGET_N:
        urea_dose_kg_ha = round(((TARGET_N - n) * 2.24) / 0.46, 1)
        deficits["Nitrogen"] = f"{round(TARGET_N - n, 1)} mg/kg deficit"
        recommendations.append(f"Urea (46% N) @ {urea_dose_kg_ha} kg/ha")
    
    if p < TARGET_P:
        dap_dose_kg_ha = round(((TARGET_P - p) * 2.24) / 0.46, 1)
        deficits["Phosphorus"] = f"{round(TARGET_P - p, 1)} mg/kg deficit"
        recommendations.append(f"DAP (18% N, 46% P2O5) @ {dap_dose_kg_ha} kg/ha")

    if k < TARGET_K:
        mop_dose_kg_ha = round(((TARGET_K - k) * 2.24) / 0.60, 1)
        deficits["Potassium"] = f"{round(TARGET_K - k, 1)} mg/kg deficit"
        recommendations.append(f"MOP (60% K2O) @ {mop_dose_kg_ha} kg/ha")

    # Timing Decision Logic
    if not recommendations:
        timing_status = "OPTIMAL_SOIL"
        timing_message = "✅ Soil nutrient profile is well-balanced. No chemical fertilizer needed."
    elif rain_prob_next_24h > 50 or moisture > 65.0:
        timing_status = "BLOCKED_LEACHING_RISK"
        timing_message = (
            f"🛑 DEFER APPLICATION: Rain probability ({rain_prob_next_24h}%) or soil moisture ({moisture}%) is too high. "
            "Applying fertilizer now will cause nutrient runoff and groundwater leaching!"
        )
    elif moisture < 20.0:
        timing_status = "BLOCKED_DRY_SOIL"
        timing_message = (
            f"⚠️ SOIL TOO DRY ({moisture}%): Fertilizer granules cannot dissolve and will scorch roots. "
            "Irrigate field first, allow water to infiltrate for 12 hours, then apply."
        )
    else:
        timing_status = "APPROVED_APPLICATION"
        timing_message = (
            f"✅ APPROVED FOR IMMEDIATE APPLICATION: Soil moisture ({moisture}%) is ideal for uptake. "
            "Apply in early morning or late afternoon to prevent evaporation loss."
        )

    return {
        "deficits": deficits,
        "fertilizers": recommendations,
        "timing_status": timing_status,
        "timing_message": timing_message
    }

if __name__ == "__main__":
    # Test sample readings matching current sensor setup
    sample_n, sample_p, sample_k = 25, 18, 60
    sample_ph, sample_m, sample_t = 6.6, 38.0, 27.5
    sample_rain = 15.0 # 15% rain forecast

    print("=" * 65)
    print("🌾 EXPERIMENTAL ADVISORY ENGINE TEST")
    print("=" * 65)
    print(f"Inputs: N={sample_n}, P={sample_p}, K={sample_k} mg/kg | pH={sample_ph} | Moisture={sample_m}% | Rain={sample_rain}%")
    
    crops = evaluate_crop_suitability(sample_n, sample_p, sample_k, sample_ph, sample_m, sample_t)
    print("\n[CROP SUITABILITY RANKINGS]:")
    for name, match in crops:
        print(f"  • {name:<20}: {match}% Match")

    fert = calculate_fertilizer_advisory(sample_n, sample_p, sample_k, sample_m, sample_rain)
    print("\n[FERTILIZER ADVISORY]:")
    for f in fert["fertilizers"]:
        print(f"  👉 Recommend: {f}")
    print(f"\n[TIMING DECISION]:\n  {fert['timing_message']}")
