# Adaptive Multimodal Edge-Cloud IoT Framework for Precision Agriculture
## Problem Statement 10: Smart Agriculture IoT Using Soil-Crop Sensor Fusion

**Course:** IoT & Industrial IoT (Publication-Oriented Engineering Project Track)  
**Target Hardware:** ESP32-WROOM-32, MAX485 Transceiver, 7-in-1 Soil RS485 Probe, Optical Crop Camera  
**Live Web Gateway:** [http://localhost:5000](http://localhost:5000)  
**Interactive Slides:** [http://localhost:5000/presentation](http://localhost:5000/presentation)  
**Technical Report:** [http://localhost:5000/report](http://localhost:5000/report) or `report.md`  

---

## 1. Executive Summary

Conventional precision agriculture relies on single-modality sensors, causing severe operational blind spots:
* **Soil-Only Systems** cannot detect foliar fungal infections, mildew, or atmospheric heat scorch.
* **Optical-Only Systems** cannot detect root-zone salinity or sub-surface NPK depletion before irreversible stunting.
* **Cloud-Only Systems** suffer from high latency ($>380\text{ ms}$), fail during rural internet dropouts, and naively irrigate right before rainstorms, causing agrochemical leaching.

This project delivers an end-to-end **Multimodal Sensor-Fusion Framework**:
1. **Underground Modality:** Physical 7-in-1 industrial soil sensor (RS485 Modbus RTU) measuring Moisture, Temp, EC, pH, N, P, K.
2. **Above-Ground Modality:** Non-contact optical crop camera computing Excess Green Index ($\text{ExG}$), canopy cover %, and chlorosis %.
3. **Edge Layer (ESP32 TinyML):** Sub-millisecond safety cutoff ($0.82\text{ ms}$) and deadband delta-filtering saving $82.5\%$ transmission bandwidth.
4. **Gateway Decision Engine:** Weighted decision fusion ($\alpha=0.55$) + Open-Meteo weather rain-gating + stoichiometric fertilizer dosage calculation.
5. **Interactive Web Dashboard:** Real-time visual dashboard at `http://localhost:5000` with live demonstration buttons.

---

## 2. 10-Fold Cross-Validation Benchmark (N = 5,000 Samples)

In strict accordance with the course guidelines, 4 competing architectures were evaluated on 5,000 samples with realistic Gaussian sensor noise:

| Architecture | Modality | 10-Fold CV Accuracy | Test Accuracy (Held-out) | Weighted F1 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Soil-Only Model** | Underground (7 parameters) | $81.4\% \pm 1.56\%$ | $82.6\%$ | $0.825$ | $0.82\text{ ms}$ (Edge) |
| **2. Crop-Only Model** | Optical (3 parameters) | $98.0\% \pm 0.67\%$ | $97.9\%$ | $0.979$ | $15.2\text{ ms}$ (CV) |
| **3. Simple Concatenation** | Early Fusion (10 parameters) | $100.0\% \pm 0.00\%$ | $100.0\%$ | $1.000$ | $385.0\text{ ms}$ (Cloud) |
| **4. Proposed Decision Fusion** | **Late Probability Fusion** | **$99.6\% \pm 0.48\%$** | **$99.4\%$** | **$0.994$** | **Sub-ms / Hybrid** |

---

## 3. Quickstart & Usage

### A. View the Live Dashboard
The web server runs automatically at:
```bash
# Open in browser:
http://localhost:5000
```
* Use the **Viva Demonstration Buttons** at the top to simulate Drought, Rain-Gating, Nutrient Deficit, or Foliar Stress in real time.
* Click **Slides ↗** or **Report ↗** in the top navigation bar.

### B. Flash the ESP32 Firmware (When Ready)
The complete firmware is located at `esp32_firmware/esp32_firmware.ino` (includes `tinyml_model.h`).
To compile and upload via Arduino CLI:
```bash
flatpak run --command=sh cc.arduino.IDE2 -c "/app/arduino-ide/resources/app/lib/backend/resources/arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 /home/deek/Projects/smart_agriculture/esp32_firmware"
```
Or simply open `/home/deek/Projects/smart_agriculture/esp32_firmware/esp32_firmware.ino` in Arduino IDE and click **Upload**.

### C. Retrain the AI Models
```bash
/home/deek/Projects/smart_agriculture/venv/bin/python3 train_multimodal_ai.py
```

---

## 4. Hardware Wiring Reference

| 7-in-1 Probe Wire | ESP32 / MAX485 Pin | Description |
| :--- | :--- | :--- |
| **Brown (VCC)** | 12V DC Output (Buck Converter) | Operating Voltage (9-24V DC) |
| **Black (GND)** | Common GND | Power Ground |
| **Yellow (A+)** | MAX485 Pin A | RS485 Differential Data + |
| **Blue (B-)** | MAX485 Pin B | RS485 Differential Data - |
| **MAX485 RO** | ESP32 GPIO 16 (RX2) | Hardware Serial2 Receive |
| **MAX485 DI** | ESP32 GPIO 17 (TX2) | Hardware Serial2 Transmit |
| **MAX485 DE+RE** | ESP32 GPIO 4 | RS485 Direction Toggle |
| **MAX485 VCC** | ESP32 3.3V / 5V | Transceiver Logic Power |

---

## 5. Repository File Map

```
/home/deek/Projects/smart_agriculture/
├── app.py                           # Master Flask Gateway Server (Port 5000)
├── ai_inference_engine.py           # Unified multimodal AI + weather decision engine
├── train_multimodal_ai.py           # 5,000-sample generator & 10-fold CV training pipeline
├── dataset.csv                      # 5,000-sample ground-truth dataset
├── model_soil.joblib                # Trained Random Forest Soil Model
├── model_crop.joblib                # Trained Random Forest Crop Model
├── model_concat.joblib              # Trained Concatenation Model
├── tinyml_model.h                   # Pure-C TinyML Decision Tree for ESP32
├── esp32_firmware/
│   ├── esp32_firmware.ino           # Verified Arduino sketch with TinyML & Deadband
│   └── tinyml_model.h               # Synchronized edge header
├── report.md                        # Publication-quality technical research paper
├── presentation.html                # 8-slide keyboard-navigable presentation deck
├── modality_accuracy_comparison.png # 4-model accuracy chart with 10-fold CV error bars
├── latency_comparison.png           # Edge (0.82 ms) vs Cloud (385 ms) speedup chart
├── bandwidth_savings.png            # Deadband delta-compression (82.5% reduction) chart
├── ai_model_comparison.png          # Precision, Recall, and F1 comparison chart
├── templates/
│   └── index.html                   # Glassmorphism live dashboard UI
└── static/
    └── latest_crop.jpg              # Live OpenCV crop camera capture
```
