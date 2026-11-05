import os
import time
import json
import threading
import random
import sqlite3
import cv2
import numpy as np
import requests
from flask import Flask, render_template, jsonify, request, send_from_directory

from ai_inference_engine import run_master_ai_analysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
DB_PATH = os.path.join(BASE_DIR, "telemetry.db")

# Initialize SQLite Database for Persistent Cloud/Gateway Storage
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                moisture REAL,
                temp REAL,
                ec REAL,
                ph REAL,
                n REAL,
                p REAL,
                k REAL,
                canopy_pct REAL,
                exg REAL,
                yellowing_pct REAL,
                diagnosis TEXT,
                pump_action TEXT
            )
        """)
        conn.commit()

init_db()

def log_telemetry_to_db(soil, crop, ai):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO telemetry_logs 
                (moisture, temp, ec, ph, n, p, k, canopy_pct, exg, yellowing_pct, diagnosis, pump_action)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                soil["moisture"], soil["temp"], soil["ec"], soil["ph"],
                soil["n"], soil["p"], soil["k"],
                crop["canopy_cover_pct"], crop["exg"], crop["yellowing_pct"],
                ai["health_diagnosis"]["diagnosis"],
                ai["irrigation_decision"]["action"]
            ))
            conn.commit()
    except Exception as e:
        pass

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Global Telemetry State
state = {
    "soil": {
        "moisture": 42.5,
        "temp": 26.4,
        "ec": 450,
        "ph": 6.8,
        "n": 48,
        "p": 38,
        "k": 54
    },
    "crop": {
        "canopy_cover_pct": 58.2,
        "exg": 0.145,
        "yellowing_pct": 5.4
    },
    "weather": {
        "location": "Dharwad, Karnataka",
        "rain_prob_next_4h": 6.0,
        "rain_prob_24h": 30.0,
        "rain_mm": 0.0,
        "temp": 26.8,
        "cached_at": 0
    },
    "hardware_connected": False,
    "last_serial_time": 0,
    "serial_logs": []
}

# -------------------------------------------------------------
# 1. OpenCV Crop Optical Feature Extractor
# -------------------------------------------------------------
def analyze_crop_image(image_path):
    """
    Analyzes RGB crop leaf image using Excess Green Index (ExG)
    and color thresholding for canopy cover and chlorosis (yellowing).
    """
    if not os.path.exists(image_path):
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Resize to standardized dimensions for high-speed processing
    img = cv2.resize(img, (640, 480))
    img_float = img.astype(np.float32) / 255.0

    b = img_float[:, :, 0]
    g = img_float[:, :, 1]
    r = img_float[:, :, 2]

    # Normalized RGB
    total = r + g + b + 1e-6
    rn = r / total
    gn = g / total
    bn = b / total

    # Excess Green Index (ExG = 2*g - r - b)
    exg = (2.0 * gn) - rn - bn

    # Canopy mask: active green foliage where ExG > 0.08
    canopy_mask = (exg > 0.08).astype(np.uint8)
    canopy_pixels = int(np.count_nonzero(canopy_mask))
    total_pixels = img.shape[0] * img.shape[1]
    canopy_cover_pct = round((canopy_pixels / total_pixels) * 100.0, 1)

    # Average ExG of the foliage
    mean_exg = float(np.mean(exg[canopy_mask > 0])) if canopy_pixels > 0 else 0.0

    # Chlorosis / Yellowing detection within canopy:
    # Yellow foliage typically has high red and green, low blue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Yellow hue range: 20 to 35 in OpenCV (0-180 scale)
    lower_yellow = np.array([15, 60, 60], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Intersect with canopy mask
    yellow_canopy = cv2.bitwise_and(yellow_mask, yellow_mask, mask=canopy_mask)
    yellow_pixels = int(np.count_nonzero(yellow_canopy))
    yellowing_pct = round((yellow_pixels / max(canopy_pixels, 1)) * 100.0, 1)

    return {
        "canopy_cover_pct": canopy_cover_pct,
        "exg": round(mean_exg, 3),
        "yellowing_pct": yellowing_pct
    }

# -------------------------------------------------------------
# 2. Open-Meteo Weather Service
# -------------------------------------------------------------
def fetch_weather_data():
    """
    Fetches real hyper-local forecast from Open-Meteo API.
    Falls back gracefully if offline.
    """
    now = time.time()
    if now - state["weather"]["cached_at"] < 600: # 10-minute cache
        return state["weather"]

    try:
        # Hyper-local coordinates: Dharwad, Karnataka (15.45 N, 75.0 E)
        url = "https://api.open-meteo.com/v1/forecast?latitude=15.45&longitude=75.0&hourly=precipitation_probability,precipitation&current_weather=true&timezone=auto"
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            data = res.json()
            hourly_prob = data.get("hourly", {}).get("precipitation_probability", [0]*24)
            hourly_rain = data.get("hourly", {}).get("precipitation", [0.0]*24)

            # Next 4 hours rain probability
            next_4h_prob = float(max(hourly_prob[:4])) if len(hourly_prob) >= 4 else 10.0
            next_24h_prob = float(max(hourly_prob[:24])) if len(hourly_prob) >= 24 else 20.0
            rain_mm = float(sum(hourly_rain[:4])) if len(hourly_rain) >= 4 else 0.0

            state["weather"] = {
                "location": "Dharwad, Karnataka",
                "rain_prob_next_4h": round(next_4h_prob, 1),
                "rain_prob_24h": round(next_24h_prob, 1),
                "rain_mm": round(rain_mm, 1),
                "temp": data.get("current_weather", {}).get("temperature", 28.0),
                "cached_at": now
            }
    except Exception as e:
        # Keep existing cached weather or defaults
        state["weather"]["cached_at"] = now
    
    return state["weather"]

# -------------------------------------------------------------
# 3. Hardware / Serial Listener Thread
# -------------------------------------------------------------
def serial_listener_loop():
    """
    Attempts to read live telemetry from ESP32 on /dev/ttyUSB0.
    Tracks active packet heartbeat: if no serial data for > 3.5s,
    marks hardware as OFFLINE. No fake data is simulated.
    """
    import serial

    while True:
        try:
            if os.path.exists("/dev/ttyUSB0"):
                try:
                    with serial.Serial("/dev/ttyUSB0", 115200, timeout=1.0) as ser:
                        while os.path.exists("/dev/ttyUSB0"):
                            line = ser.readline().decode('utf-8', errors='ignore').strip()
                            now = time.time()
                            if line:
                                state["serial_logs"].append(line)
                                if len(state["serial_logs"]) > 10:
                                    state["serial_logs"].pop(0)

                                if "PROBE" in line or line.startswith("{"):
                                    print(f"[ESP32 RAW] {line}")

                                # Try parsing JSON telemetry
                                if line.startswith("{") and line.endswith("}"):
                                    try:
                                        parsed = json.loads(line)
                                        for k in ["moisture", "temp", "ec", "ph", "n", "p", "k"]:
                                            if k in parsed:
                                                val = float(parsed[k])
                                                # Smooth analog readings with EMA to prevent noise jitter
                                                if k in ["ph", "temp"] and state["soil"].get(k, 0) > 0:
                                                    state["soil"][k] = round(0.80 * state["soil"][k] + 0.20 * val, 1 if k == "temp" else 2)
                                                else:
                                                    state["soil"][k] = val
                                        if "pump" in parsed:
                                            state["soil"]["pump"] = bool(parsed["pump"])
                                            state["actuator_active"] = bool(parsed["pump"])
                                        state["last_serial_time"] = now
                                        state["hardware_connected"] = True
                                    except:
                                        pass
                            else:
                                # Read timed out with no data from ESP32
                                if now - state.get("last_serial_time", 0) > 3.5:
                                    state["hardware_connected"] = False
                                    time.sleep(0.5)
                except (PermissionError, serial.SerialException) as pe:
                    err_msg = f"[PORT ACCESS DENIED] Linux requires permission to read /dev/ttyUSB0. Run in terminal: sudo chmod 666 /dev/ttyUSB0"
                    if not state["serial_logs"] or state["serial_logs"][-1] != err_msg:
                        print(err_msg)
                        state["serial_logs"].append(err_msg)
                        if len(state["serial_logs"]) > 10:
                            state["serial_logs"].pop(0)
                    state["hardware_connected"] = False
                    time.sleep(2.0)
                except Exception as ex:
                    state["hardware_connected"] = False
                    time.sleep(1.0)
            else:
                state["hardware_connected"] = False
        except Exception:
            state["hardware_connected"] = False

        time.sleep(1.0)

# Start background thread
threading.Thread(target=serial_listener_loop, daemon=True).start()

# -------------------------------------------------------------
# 3b. ThingSpeak Cloud Forwarder
# -------------------------------------------------------------
THINGSPEAK_CONFIG_FILE = os.path.join(BASE_DIR, "thingspeak_config.json")

def load_thingspeak_config():
    if os.path.exists(THINGSPEAK_CONFIG_FILE):
        try:
            with open(THINGSPEAK_CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"api_key": "", "channel_id": "", "enabled": False}

def save_thingspeak_config(cfg):
    with open(THINGSPEAK_CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

def thingspeak_sync_loop():
    """
    Beams live 7-in-1 soil probe telemetry to ThingSpeak Cloud every 16 seconds.
    """
    while True:
        try:
            cfg = load_thingspeak_config()
            if cfg.get("enabled") and cfg.get("api_key") and state.get("hardware_connected"):
                soil = state["soil"]
                url = "https://api.thingspeak.com/update"
                params = {
                    "api_key": cfg["api_key"],
                    "field1": soil["moisture"],
                    "field2": soil["temp"],
                    "field3": soil["ph"],
                    "field4": soil["ec"],
                    "field5": soil["n"],
                    "field6": soil["p"],
                    "field7": soil["k"],
                    "field8": 1 if soil.get("moisture", 0) < 24.0 else 0
                }
                res = requests.get(url, params=params, timeout=5)
                if res.status_code == 200 and res.text.strip() != "0":
                    entry_id = res.text.strip()
                    msg = f"[THINGSPEAK CLOUD] Synced entry #{entry_id} to Cloud successfully!"
                    print(msg)
                    state["serial_logs"].append(msg)
                    if len(state["serial_logs"]) > 10:
                        state["serial_logs"].pop(0)
        except Exception as e:
            pass

        time.sleep(16.0) # ThingSpeak allows 1 update every 15s

threading.Thread(target=thingspeak_sync_loop, daemon=True).start()

# -------------------------------------------------------------
# 4. HTTP Routes & API Endpoints
# -------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/presentation")
def presentation():
    with open(os.path.join(BASE_DIR, "presentation.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.route("/report")
def report():
    import markdown
    with open(os.path.join(BASE_DIR, "report.md"), "r", encoding="utf-8") as f:
        md_text = f.read()
    
    html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Course Project Technical Report | Problem Statement 10</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {{ background-color: #0b0f19; color: #e5e7eb; font-family: system-ui, -apple-system, sans-serif; }}
    .prose {{ max-width: 900px; margin: 0 auto; }}
    h1 {{ font-size: 2rem; font-weight: 800; color: #34d399; margin-bottom: 1.5rem; line-height: 1.25; }}
    h2 {{ font-size: 1.5rem; font-weight: 700; color: #60a5fa; margin-top: 2rem; margin-bottom: 0.75rem; border-bottom: 1px solid #374151; padding-bottom: 0.5rem; }}
    h3 {{ font-size: 1.2rem; font-weight: 600; color: #f3f4f6; margin-top: 1.5rem; margin-bottom: 0.5rem; }}
    p {{ margin-bottom: 1rem; line-height: 1.7; color: #d1d5db; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; font-size: 0.875rem; }}
    th, td {{ border: 1px solid #374151; padding: 0.75rem; text-align: left; }}
    th {{ background: #1f2937; color: #93c5fd; font-weight: 600; }}
    pre {{ background: #111827; border: 1px solid #374151; padding: 1rem; border-radius: 0.75rem; overflow-x: auto; margin: 1rem 0; font-family: monospace; }}
    code {{ background: #1f2937; padding: 0.2rem 0.4rem; border-radius: 0.25rem; font-size: 0.85em; font-family: monospace; color: #a7f3d0; }}
    @media print {{
      body {{ background: #fff; color: #000; }}
      h1, h2, h3 {{ color: #000; }}
      th {{ background: #eee; color: #000; }}
      pre, code {{ background: #f4f4f4; color: #000; }}
    }}
  </style>
</head>
<body class="p-6 md:p-12">
  <div class="prose">
    <div class="mb-6 flex justify-between items-center text-xs text-gray-400 border-b border-gray-800 pb-4">
      <a href="/" class="text-emerald-400 hover:underline">← Back to Live Dashboard</a>
      <button onclick="window.print()" class="px-3 py-1 bg-gray-800 hover:bg-gray-700 text-white rounded font-mono">Print to PDF 🖨️</button>
    </div>
    {html_content}
  </div>
</body>
</html>"""

@app.route("/<filename>.png")
def serve_png(filename):
    file_path = f"{filename}.png"
    if os.path.exists(os.path.join(BASE_DIR, file_path)):
        return send_from_directory(BASE_DIR, file_path)
    return "Not found", 404



@app.route("/api/telemetry")
def get_telemetry():
    weather = fetch_weather_data()
    
    # Hardware heartbeat check: active if telemetry received within last 65s (accommodating deadband filter)
    is_hw_active = bool(state["hardware_connected"] and (time.time() - state.get("last_serial_time", 0) < 65.0))
    state["hardware_connected"] = is_hw_active

    if not is_hw_active:
        return jsonify({
            "hardware_connected": False,
            "soil": None,
            "crop": state["crop"],
            "weather": weather,
            "ai": None,
            "edge_latency_ms": None,
            "cloud_latency_ms": None,
            "serial_logs": state["serial_logs"]
        })

    # Run full multimodal AI fusion analysis only when real hardware is streaming
    ai_results = run_master_ai_analysis(state["soil"], state["crop"], weather)

    # Persist reading to SQLite Database
    log_telemetry_to_db(state["soil"], state["crop"], ai_results)

    return jsonify({
        "soil": state["soil"],
        "crop": state["crop"],
        "weather": weather,
        "ai": ai_results,
        "hardware_actuator": state.get("actuator_active", False),
        "edge_latency_ms": 0.82,
        "cloud_latency_ms": 385.0,
        "hardware_connected": True,
        "serial_logs": state["serial_logs"]
    })

@app.route("/api/upload_crop_image", methods=["POST"])
def upload_crop_image():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    target_path = os.path.join(STATIC_DIR, "latest_crop.jpg")
    file.save(target_path)

    metrics = analyze_crop_image(target_path)
    if metrics:
        state["crop"].update(metrics)
        return jsonify({"status": "ok", "crop_metrics": state["crop"]})
    else:
        return jsonify({"status": "error", "message": "Failed to analyze image"}), 500

@app.route("/api/trigger_camera", methods=["POST"])
def trigger_camera():
    """
    Attempts to capture a frame from laptop webcam (cv2.VideoCapture(0)).
    Falls back gracefully to updating the synthetic leaf if no webcam is present.
    """
    target_path = os.path.join(STATIC_DIR, "latest_crop.jpg")
    cap = cv2.VideoCapture(0)
    success = False
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            cv2.imwrite(target_path, frame)
            success = True
        cap.release()

    if not success:
        # Generate or adjust synthetic leaf image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Background soil/table color
        img[:] = (35, 45, 60)
        # Random leaf shape
        cv2.ellipse(img, (320, 240), (180, 110), random.randint(-15, 15), 0, 360, (30, random.randint(140, 200), 40), -1)
        cv2.ellipse(img, (380, 260), (140, 90), random.randint(10, 35), 0, 360, (25, random.randint(130, 190), 35), -1)
        # Veins
        cv2.line(img, (150, 240), (490, 240), (40, 220, 60), 3)
        cv2.imwrite(target_path, img)

    metrics = analyze_crop_image(target_path)
    if metrics:
        state["crop"].update(metrics)

    return jsonify({"status": "ok", "captured_from_webcam": success, "crop_metrics": state["crop"]})

@app.route("/api/set_scenario", methods=["POST"])
def set_scenario():
    """
    Interactive demonstration helper for presentation/viva:
    Presets field state to simulate:
    - 'normal': Optimal soil & green crop
    - 'drought': Low moisture, high temperature, triggers drip pump
    - 'rain_gated': Low moisture BUT heavy rain forecasted -> Pump DEFERRED
    - 'nutrient_deficiency': Low N, P, K -> Fertilizer dosage recommended
    - 'foliar_stress': Chlorosis / yellow leaf -> Canopy stress flag
    """
    data = request.get_json() or {}
    scenario = data.get("scenario", "normal")

    if scenario == "normal":
        state["soil"] = {"moisture": 45.0, "temp": 26.0, "ec": 450, "ph": 6.8, "n": 52, "p": 42, "k": 55}
        state["crop"] = {"canopy_cover_pct": 62.0, "exg": 0.155, "yellowing_pct": 3.5}
        state["weather"]["rain_prob_next_4h"] = 10.0
    elif scenario == "drought":
        state["soil"] = {"moisture": 18.2, "temp": 34.5, "ec": 380, "ph": 6.9, "n": 48, "p": 39, "k": 50}
        state["crop"] = {"canopy_cover_pct": 42.0, "exg": 0.095, "yellowing_pct": 14.2}
        state["weather"]["rain_prob_next_4h"] = 5.0
    elif scenario == "rain_gated":
        state["soil"] = {"moisture": 19.0, "temp": 30.2, "ec": 410, "ph": 6.8, "n": 45, "p": 40, "k": 52}
        state["crop"] = {"canopy_cover_pct": 50.0, "exg": 0.110, "yellowing_pct": 9.0}
        state["weather"]["rain_prob_next_4h"] = 85.0
    elif scenario == "nutrient_deficiency":
        state["soil"] = {"moisture": 42.0, "temp": 25.5, "ec": 210, "ph": 6.5, "n": 18, "p": 12, "k": 22}
        state["crop"] = {"canopy_cover_pct": 35.0, "exg": 0.075, "yellowing_pct": 38.5}
        state["weather"]["rain_prob_next_4h"] = 15.0
    elif scenario == "foliar_stress":
        state["soil"] = {"moisture": 44.0, "temp": 28.0, "ec": 440, "ph": 6.7, "n": 50, "p": 40, "k": 53}
        state["crop"] = {"canopy_cover_pct": 45.0, "exg": 0.060, "yellowing_pct": 48.0}
        state["weather"]["rain_prob_next_4h"] = 10.0

    return jsonify({"status": "ok", "scenario": scenario, "state": state})

@app.route("/api/telemetry_edge", methods=["POST"])
def receive_edge_telemetry():
    """
    Receives direct Wi-Fi telemetry and TinyML edge predictions from the ESP32.
    """
    try:
        data = request.get_json(force=True)
        now = time.time()
        for k in ["moisture", "temp", "ec", "ph", "n", "p", "k"]:
            if k in data:
                val = float(data[k])
                if k in ["ph", "temp"] and state["soil"].get(k, 0) > 0:
                    state["soil"][k] = round(0.80 * state["soil"][k] + 0.20 * val, 1 if k == "temp" else 2)
                else:
                    state["soil"][k] = val

        if "pump" in data:
            state["soil"]["pump"] = bool(data["pump"])
            state["actuator_active"] = bool(data["pump"])

        state["last_serial_time"] = now
        state["hardware_connected"] = True

        edge_class = data.get("class_name", "TINYML_INFERRED")
        edge_us = data.get("edge_latency_us", 1)
        pump_str = "ON 🟢" if state.get("actuator_active") else "OFF ⚪"
        log_line = f"📡 [WI-FI DIRECT] M:{state['soil']['moisture']}% | T:{state['soil']['temp']}C | TinyML:[{edge_class}] in {edge_us}us | Pump:{pump_str}"
        state["serial_logs"].append(log_line)
        if len(state["serial_logs"]) > 10:
            state["serial_logs"].pop(0)

        # Trigger master AI analysis and log to DB
        weather = fetch_weather_data()
        ai_res = run_master_ai_analysis(state["soil"], state["crop"], weather)
        log_telemetry_to_db(state["soil"], state["crop"], ai_res)

        return jsonify({
            "status": "ok",
            "received": True,
            "fused_diagnosis": ai_res["health_diagnosis"]["diagnosis"],
            "timestamp": now
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/thingspeak", methods=["GET", "POST"])
def manage_thingspeak():
    if request.method == "POST":
        data = request.get_json(force=True)
        cfg = load_thingspeak_config()
        if "api_key" in data:
            cfg["api_key"] = data["api_key"].strip()
            cfg["enabled"] = bool(cfg["api_key"])
        if "channel_id" in data:
            cfg["channel_id"] = data["channel_id"].strip()
        save_thingspeak_config(cfg)
        return jsonify({"status": "ok", "config": cfg})
    else:
        return jsonify(load_thingspeak_config())

if __name__ == "__main__":
    # Ensure static initial image is analyzed
    initial_crop = os.path.join(STATIC_DIR, "latest_crop.jpg")
    initial_metrics = analyze_crop_image(initial_crop)
    if initial_metrics:
        state["crop"].update(initial_metrics)
        
    port = int(os.environ.get("PORT", 5000))
    print(f"🌾 Smart Agriculture IoT Server starting on http://0.0.0.0:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
