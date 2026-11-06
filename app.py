import os
import time
import datetime
import json
import threading
import random
import sqlite3
import tempfile
import fcntl
from email.utils import parsedate_to_datetime
import cv2
import numpy as np
import requests
from flask import Flask, render_template, jsonify, request, send_from_directory, Response, send_file

from ai_inference_engine import (
    run_master_ai_analysis,
    evaluate_foliar_disease,
    CROP_BENCHMARKS,
    analyze_crop_deficiencies
)

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

# Fallback 5-Day Agricultural Forecast Generator (Dharwad, Karnataka)
def generate_fallback_5day():
    days = []
    base_date = datetime.datetime.now()
    weather_pattern = [
        ("Drizzle", "fa-cloud-rain text-sky-500", 30.7, 21.9, 1.4),
        ("Drizzle", "fa-cloud-rain text-sky-500", 30.2, 20.9, 2.4),
        ("Light Rain", "fa-cloud-rain text-sky-500", 30.4, 20.8, 0.6),
        ("Partly Cloudy", "fa-cloud-sun text-amber-500", 30.8, 20.6, 0.0),
        ("Clear Sky", "fa-sun text-amber-500", 30.4, 19.8, 0.0)
    ]
    for i in range(1, 6):
        d = base_date + datetime.timedelta(days=i)
        p = weather_pattern[i-1]
        days.append({
            "date": d.strftime("%Y-%m-%d"),
            "day_name": d.strftime("%a"),
            "day_date": d.strftime("%b %d"),
            "condition": p[0],
            "icon": p[1],
            "temp_max": p[2],
            "temp_min": p[3],
            "rain_mm": p[4]
        })
    return days

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
        "condition": "Clear Sky",
        "icon": "fa-sun text-amber-500",
        "is_day": True,
        "temp": 24.9,
        "humidity": 65,
        "wind_kmh": 10.8,
        "rain_mm_next_4h": 0.0,
        "rain_mm_next_24h": 0.0,
        "rain_mm": 0.0,
        "daily_forecast": generate_fallback_5day(),
        "cached_at": 0,
        "provider": "Fallback estimate",
        "live_fetch": False,
        "stale": True
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

    # HSV & Grayscale representation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Plant foliage mask (green leaves, chlorotic yellow foliage, and green stems)
    plant_mask = (((hsv[:, :, 1] > 25) & (hsv[:, :, 2] > 30)) | (exg > 0.02)).astype(np.uint8)
    plant_pixels = int(np.count_nonzero(plant_mask))
    total_pixels = img.shape[0] * img.shape[1]
    canopy_cover_pct = round((plant_pixels / total_pixels) * 100.0, 1)

    # Average ExG of the plant canopy
    mean_exg = float(np.mean(exg[plant_mask > 0])) if plant_pixels > 0 else 0.0

    # Chlorosis (Yellowing) within plant foliage (Hue 18-31: true yellow/chlorotic tissue):
    lower_yellow = np.array([18, 55, 55], dtype=np.uint8)
    upper_yellow = np.array([31, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_plant = cv2.bitwise_and(yellow_mask, yellow_mask, mask=plant_mask)
    yellow_pixels = int(np.count_nonzero(yellow_plant))
    yellowing_pct = round((yellow_pixels / max(plant_pixels, 1)) * 100.0, 1)

    # Dark necrotic lesions or insect clusters:
    # In vigorous green crops (mean_exg >= 0.20), dark regions are natural canopy shadows.
    # On low-vigor or chlorotic foliage (mean_exg < 0.20), dark regions indicate necrotic lesions or pest colonies.
    if mean_exg >= 0.20:
        dark_spot_pct = 0.0
    else:
        dark_mask = ((gray < 70) & (plant_mask > 0)).astype(np.uint8)
        dark_pixels = int(np.count_nonzero(dark_mask))
        dark_spot_pct = round((dark_pixels / max(plant_pixels, 1)) * 100.0, 1)

    return {
        "canopy_cover_pct": canopy_cover_pct,
        "exg": round(mean_exg, 3),
        "yellowing_pct": yellowing_pct,
        "dark_spot_pct": dark_spot_pct
    }

# -------------------------------------------------------------
# 2. Cached weather service (MET Norway)
# -------------------------------------------------------------
WEATHER_CACHE_FILE = os.path.join(BASE_DIR, "weather_cache.json")
WEATHER_LOCK_FILE = WEATHER_CACHE_FILE + ".lock"
WEATHER_MIN_REFRESH_SECONDS = 6 * 60 * 60

def parse_met_symbol(symbol_code):
    if not symbol_code:
        return "Fair", "fa-cloud-sun text-amber-500"
    base = symbol_code.lower()
    is_night = "_night" in base
    base = base.replace("_day", "").replace("_night", "").replace("_polartwilight", "")
    mapping = {
        "clearsky": ("Clear Sky" if not is_night else "Clear Night", "fa-sun text-amber-500" if not is_night else "fa-moon text-sky-400"),
        "fair": ("Fair", "fa-cloud-sun text-amber-500" if not is_night else "fa-cloud-moon text-slate-400"),
        "partlycloudy": ("Partly Cloudy", "fa-cloud-sun text-amber-500" if not is_night else "fa-cloud-moon text-slate-400"),
        "cloudy": ("Overcast", "fa-cloud text-slate-500"),
        "fog": ("Foggy", "fa-smog text-slate-400"),
        "lightrainshowers": ("Light Showers", "fa-cloud-sun-rain text-sky-400" if not is_night else "fa-cloud-moon-rain text-sky-400"),
        "rainshowers": ("Rain Showers", "fa-cloud-sun-rain text-sky-500" if not is_night else "fa-cloud-moon-rain text-sky-500"),
        "heavyrainshowers": ("Heavy Showers", "fa-cloud-showers-heavy text-sky-600"),
        "lightrain": ("Light Rain", "fa-cloud-rain text-sky-500"),
        "rain": ("Rain", "fa-cloud-rain text-sky-600"),
        "heavyrain": ("Heavy Rain", "fa-cloud-showers-heavy text-sky-600"),
        "lightrainandthunder": ("Thunder Drizzle", "fa-cloud-bolt text-amber-600"),
        "rainandthunder": ("Thunderstorm", "fa-bolt text-amber-600"),
        "heavyrainandthunder": ("Heavy Thunderstorm", "fa-bolt text-amber-600"),
    }
    return mapping.get(base, ("Fair", "fa-cloud-sun text-amber-500"))

def _fetch_from_met_no(cached=None):
    """Fetch MET Norway data, conditionally when its Last-Modified value is known."""
    from collections import defaultdict
    url = "https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=15.45&lon=75.00"
    headers = {"User-Agent": "VerdantFarm/1.0 (+https://smart-agriculture-iot-o3ni.onrender.com)"}
    if cached and cached.get("last_modified"):
        headers["If-Modified-Since"] = cached["last_modified"]
    res = requests.get(url, headers=headers, timeout=12)
    if res.status_code == 304 and cached and cached.get("weather"):
        return cached["weather"], res.headers, True
    if res.status_code != 200:
        raise RuntimeError(f"MET Norway returned HTTP {res.status_code}")
    data = res.json()
    ts = data.get("properties", {}).get("timeseries", [])
    if not ts:
        raise ValueError("MET Norway returned empty timeseries")

    cur_entry = ts[0]
    inst = cur_entry.get("data", {}).get("instant", {}).get("details", {})
    temp_c = round(float(inst.get("air_temperature", 25.0)), 1)
    humidity = int(inst.get("relative_humidity", 60))
    wind_kmh = round(float(inst.get("wind_speed", 3.0)) * 3.6, 1)

    cur_sym = cur_entry.get("data", {}).get("next_1_hours", {}).get("summary", {}).get("symbol_code") or \
              cur_entry.get("data", {}).get("next_6_hours", {}).get("summary", {}).get("symbol_code", "fair_day")
    condition, icon = parse_met_symbol(cur_sym)
    is_day = "_night" not in cur_sym

    # MET Norway does not provide precipitation probability for Dharwad.
    # Report forecast amounts in mm rather than inventing a percentage.
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    next_4h_rains = []
    next_24h_rains = []
    for entry in ts:
        try:
            entry_time = datetime.datetime.fromisoformat(entry["time"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        delta_hours = (entry_time - now_utc).total_seconds() / 3600
        data_periods = entry.get("data", {})
        amount = None
        for period in ("next_1_hours", "next_6_hours", "next_12_hours"):
            details = data_periods.get(period, {}).get("details", {})
            if "precipitation_amount" in details:
                amount = float(details["precipitation_amount"])
                break
        if amount is not None and -1 <= delta_hours < 4:
            next_4h_rains.append(amount)
        if amount is not None and -1 <= delta_hours < 24:
            next_24h_rains.append(amount)
    rain_mm_4h = round(sum(next_4h_rains), 1)
    rain_mm_24h = round(sum(next_24h_rains), 1)

    # Next 5-Day Agricultural Forecast Aggregation
    days_data = defaultdict(lambda: {"temps": [], "precip": 0.0, "symbols": []})
    for entry in ts:
        dt_str = entry["time"][:10]
        t = entry.get("data", {}).get("instant", {}).get("details", {}).get("air_temperature")
        if t is not None:
            days_data[dt_str]["temps"].append(t)
        for k in ["next_1_hours", "next_6_hours", "next_12_hours"]:
            period_data = entry.get("data", {}).get(k)
            if period_data:
                p = period_data.get("details", {}).get("precipitation_amount")
                if p is not None:
                    days_data[dt_str]["precip"] += float(p)
                s = period_data.get("summary", {}).get("symbol_code")
                if s:
                    days_data[dt_str]["symbols"].append(s)
                break

    today_local = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    forecast_5day = []
    for d_str in sorted(days_data.keys()):
        if d_str <= today_local:
            continue
        info = days_data[d_str]
        if not info["temps"]:
            continue
        dt = datetime.datetime.strptime(d_str, "%Y-%m-%d")
        syms = [s for s in info["symbols"] if "_night" not in s]
        rep_sym = syms[len(syms)//2] if syms else (info["symbols"][0] if info["symbols"] else "fair_day")
        c_text, c_icon = parse_met_symbol(rep_sym)
        rain_mm = round(info["precip"], 1)
        forecast_5day.append({
            "date": d_str,
            "day_name": dt.strftime("%a"),
            "day_date": dt.strftime("%b %d"),
            "temp_max": round(max(info["temps"]), 1),
            "temp_min": round(min(info["temps"]), 1),
            "condition": c_text,
            "icon": c_icon,
            "rain_mm": rain_mm
        })
        if len(forecast_5day) == 5:
            break

    return {
        "location": "Dharwad, Karnataka",
        "condition": condition,
        "icon": icon,
        "is_day": is_day,
        "temp": temp_c,
        "humidity": humidity,
        "wind_kmh": wind_kmh,
        "rain_mm_next_4h": rain_mm_4h,
        "rain_mm_next_24h": rain_mm_24h,
        "rain_mm": rain_mm_4h,
        "daily_forecast": forecast_5day or generate_fallback_5day(),
        "cached_at": time.time(),
        "live_fetch": True,
        "provider": "MET Norway",
        "provider_url": "api.met.no"
    }, res.headers, False

def fetch_weather_data():
    """Use a cross-worker cache and a six-hour request floor for MET Norway."""
    now = time.time()
    if now < state["weather"].get("refresh_after", 0):
        return state["weather"]
    with open(WEATHER_LOCK_FILE, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        disk_cache = {}
        try:
            with open(WEATHER_CACHE_FILE, "r") as cache_file:
                disk_cache = json.load(cache_file)
        except (OSError, ValueError):
            pass
        if disk_cache.get("weather") and now < disk_cache.get("refresh_after", 0):
            state["weather"].update(disk_cache["weather"])
            state["weather"]["refresh_after"] = disk_cache["refresh_after"]
            return state["weather"]

        try:
            weather_result, response_headers, not_modified = _fetch_from_met_no(disk_cache)
            checked_at = time.time()
            expires_at = checked_at
            expires_header = response_headers.get("Expires")
            if expires_header:
                try:
                    expires_at = parsedate_to_datetime(expires_header).timestamp()
                except (TypeError, ValueError, OverflowError):
                    pass
            refresh_after = max(expires_at, checked_at + WEATHER_MIN_REFRESH_SECONDS)
            last_modified = response_headers.get("Last-Modified") or disk_cache.get("last_modified")
            weather_result.update({
                "cached_at": disk_cache.get("weather", {}).get("cached_at", checked_at) if not_modified else checked_at,
                "checked_at": checked_at,
                "refresh_after": refresh_after,
                "live_fetch": True,
                "stale": False
            })
            envelope = {"weather": weather_result, "refresh_after": refresh_after, "last_modified": last_modified}
            directory = os.path.dirname(WEATHER_CACHE_FILE)
            with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, encoding="utf-8") as temp_file:
                json.dump(envelope, temp_file)
                temp_path = temp_file.name
            os.replace(temp_path, WEATHER_CACHE_FILE)
            state["weather"].update(weather_result)
            print(f"[WEATHER] MET Norway refreshed: {weather_result['temp']}°C; next refresh after {datetime.datetime.fromtimestamp(refresh_after).isoformat(timespec='minutes')}")
            return state["weather"]
        except Exception as exc:
            print(f"[WEATHER] MET Norway failed: {exc}")
            if disk_cache.get("weather"):
                stale_weather = disk_cache["weather"]
                stale_weather.update({"stale": True, "error": str(exc), "refresh_after": time.time() + 3600})
                state["weather"].update(stale_weather)
                disk_cache["refresh_after"] = stale_weather["refresh_after"]
                try:
                    with open(WEATHER_CACHE_FILE, "w") as cache_file:
                        json.dump(disk_cache, cache_file)
                except OSError:
                    pass
                return state["weather"]
            fallback = state["weather"]
            retry_after = time.time() + 3600
            fallback.update({"stale": True, "error": str(exc), "provider": "Offline fallback", "refresh_after": retry_after})
            try:
                with open(WEATHER_CACHE_FILE, "w") as cache_file:
                    json.dump({"weather": fallback, "refresh_after": retry_after}, cache_file)
            except OSError:
                pass
            return fallback

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
      <div class="flex items-center gap-3">
        <a href="/report.pdf" class="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded font-mono font-bold transition flex items-center gap-1.5">📥 Download PDF Report</a>
        <button onclick="window.print()" class="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-white rounded font-mono">Print 🖨️</button>
      </div>
    </div>
    {html_content}
  </div>
</body>
</html>"""

@app.route("/report.pdf")
def serve_report_pdf():
    pdf_path = os.path.join(BASE_DIR, "Smart_Agriculture_IoT_Research_Report.pdf")
    if os.path.exists(pdf_path):
        return send_file(pdf_path, mimetype='application/pdf', as_attachment=False)
    return "PDF not found", 404

@app.route("/<filename>.png")
def serve_png(filename):
    file_path = f"{filename}.png"
    if os.path.exists(os.path.join(BASE_DIR, file_path)):
        return send_from_directory(BASE_DIR, file_path)
    return "Not found", 404

@app.route("/<filename>.jpg")
def serve_jpg(filename):
    file_path = f"{filename}.jpg"
    if os.path.exists(os.path.join(BASE_DIR, file_path)):
        return send_from_directory(BASE_DIR, file_path)
    return "Not found", 404



@app.route("/api/version")
def get_version():
    return jsonify({
        "version": "v1.4-met-weather-cached",
        "timestamp": time.time(),
        "daily_count": len(state["weather"].get("daily_forecast", [])),
        "weather": state["weather"]
    })

@app.route("/api/weather_debug")
def weather_debug():
    """Show cached provider status without making a separate API request."""
    weather = fetch_weather_data()
    return jsonify({
        "timestamp": time.time(),
        "ok": bool(weather.get("live_fetch")),
        "provider": weather.get("provider"),
        "cached_weather": weather,
        "next_refresh_at": weather.get("refresh_after")
    })

@app.route("/api/telemetry")
def get_telemetry():
    weather = fetch_weather_data()
    
    # Hardware heartbeat check: active if telemetry received within last 65s (accommodating deadband filter)
    is_hw_active = bool(state["hardware_connected"] and (time.time() - state.get("last_serial_time", 0) < 65.0))
    state["hardware_connected"] = is_hw_active

    if not is_hw_active:
        foliar_diag = evaluate_foliar_disease(state["crop"])
        y_pct = float(state["crop"].get("yellowing_pct", 0.0))
        d_pct = float(state["crop"].get("dark_spot_pct", 0.0))
        foliar_spray = foliar_diag.get("foliar_spray", "🌿 Canopy healthy. No emergency foliar spray required.")
        return jsonify({
            "hardware_connected": False,
            "soil": None,
            "crop": state["crop"],
            "weather": weather,
            "ai": {
                "health_diagnosis": foliar_diag,
                "fertilizer_advisory": {
                    "fertilizers": [],
                    "foliar_spray": foliar_spray,
                    "has_foliar_alert": bool(y_pct > 15.0 or d_pct >= 15.0),
                    "timing_status": "SOIL_SENSOR_OFFLINE",
                    "timing_message": "Soil probe offline. Non-contact optical foliar diagnosis active."
                },
                "crop_rankings": [],
                "irrigation_decision": {
                    "action": "OFFLINE",
                    "reason": "Soil probe offline."
                }
            },
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
        state["crop"] = {"canopy_cover_pct": 62.0, "exg": 0.155, "yellowing_pct": 3.5, "dark_spot_pct": 1.5}
        state["weather"]["rain_mm_next_4h"] = 0.0
        state["weather"]["rain_mm_next_24h"] = 0.0
    elif scenario == "drought":
        state["soil"] = {"moisture": 18.2, "temp": 34.5, "ec": 380, "ph": 6.9, "n": 48, "p": 39, "k": 50}
        state["crop"] = {"canopy_cover_pct": 42.0, "exg": 0.095, "yellowing_pct": 14.2, "dark_spot_pct": 3.0}
        state["weather"]["rain_mm_next_4h"] = 0.0
        state["weather"]["rain_mm_next_24h"] = 0.0
    elif scenario == "rain_gated":
        state["soil"] = {"moisture": 19.0, "temp": 30.2, "ec": 410, "ph": 6.8, "n": 45, "p": 40, "k": 52}
        state["crop"] = {"canopy_cover_pct": 50.0, "exg": 0.110, "yellowing_pct": 9.0, "dark_spot_pct": 2.0}
        state["weather"]["rain_mm_next_4h"] = 12.5
        state["weather"]["rain_mm_next_24h"] = 12.5
        state["weather"]["rain_mm"] = 12.5
        state["weather"]["condition"] = "Rain Showers"
        state["weather"]["icon"] = "fa-cloud-showers-heavy text-sky-600"
    elif scenario == "nutrient_deficiency":
        state["soil"] = {"moisture": 42.0, "temp": 25.5, "ec": 210, "ph": 6.5, "n": 18, "p": 12, "k": 22}
        state["crop"] = {"canopy_cover_pct": 35.0, "exg": 0.075, "yellowing_pct": 35.0, "dark_spot_pct": 4.0}
        state["weather"]["rain_mm_next_4h"] = 0.0
        state["weather"]["rain_mm_next_24h"] = 0.0
    elif scenario == "foliar_stress":
        state["soil"] = {"moisture": 44.0, "temp": 28.0, "ec": 440, "ph": 6.7, "n": 50, "p": 40, "k": 53}
        state["crop"] = {"canopy_cover_pct": 64.4, "exg": 0.174, "yellowing_pct": 33.2, "dark_spot_pct": 19.2}
        state["weather"]["rain_mm_next_4h"] = 0.0
        state["weather"]["rain_mm_next_24h"] = 0.0
    elif scenario == "aphids":
        state["soil"] = {"moisture": 45.0, "temp": 27.0, "ec": 460, "ph": 6.8, "n": 50, "p": 42, "k": 54}
        state["crop"] = {"canopy_cover_pct": 52.3, "exg": 0.181, "yellowing_pct": 8.0, "dark_spot_pct": 25.5}
        state["weather"]["rain_mm_next_4h"] = 0.0
        state["weather"]["rain_mm_next_24h"] = 0.0

    state["hardware_connected"] = True
    state["last_serial_time"] = time.time()

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

@app.route("/api/crop_planning", methods=["GET"])
def api_crop_planning():
    crop_name = request.args.get("crop", "Wheat")
    if crop_name not in CROP_BENCHMARKS:
        crop_name = "Wheat"
    b = CROP_BENCHMARKS[crop_name]

    # Hardware heartbeat check: active if telemetry received within last 65s
    is_hw_active = bool(state["hardware_connected"] and (time.time() - state.get("last_serial_time", 0) < 65.0) and state.get("soil"))
    state["hardware_connected"] = is_hw_active

    if not is_hw_active:
        return jsonify({
            "hardware_connected": False,
            "crop": crop_name,
            "category": b["category"],
            "icon": b["icon"],
            "description": b["desc"],
            "readiness_pct": None,
            "optimal_count": 0,
            "total_params": 7,
            "params": [
                {"name": "Soil Moisture", "unit": "%", "key": "moisture", "current": None, "min": b["min_m"], "max": b["max_m"], "target": b["target_m"], "status": "OFFLINE", "diff": None},
                {"name": "Nitrogen (N)", "unit": "mg/kg", "key": "n", "current": None, "min": b["min_n"], "max": b["max_n"], "target": b["target_n"], "status": "OFFLINE", "diff": None},
                {"name": "Phosphorus (P)", "unit": "mg/kg", "key": "p", "current": None, "min": b["min_p"], "max": b["max_p"], "target": b["target_p"], "status": "OFFLINE", "diff": None},
                {"name": "Potassium (K)", "unit": "mg/kg", "key": "k", "current": None, "min": b["min_k"], "max": b["max_k"], "target": b["target_k"], "status": "OFFLINE", "diff": None},
                {"name": "Soil pH", "unit": "pH", "key": "ph", "current": None, "min": b["min_ph"], "max": b["max_ph"], "target": b["target_ph"], "status": "OFFLINE", "diff": None},
                {"name": "Salinity (EC)", "unit": "µS/cm", "key": "ec", "current": None, "min": b["min_ec"], "max": b["max_ec"], "target": b["target_ec"], "status": "OFFLINE", "diff": None},
                {"name": "Soil Temp", "unit": "°C", "key": "temp", "current": None, "min": b["min_temp"], "max": b["max_temp"], "target": round((b["min_temp"]+b["max_temp"])/2.0, 1), "status": "OFFLINE", "diff": None}
            ],
            "remediations": [
                f"⚠️ Field sensor is disconnected. Connect ESP32 soil probe in Field Block A-12 to evaluate real-time soil readiness for {crop_name}."
            ],
            "available_crops": list(CROP_BENCHMARKS.keys())
        })

    soil = state["soil"]
    analysis = analyze_crop_deficiencies(crop_name, soil)
    analysis["available_crops"] = list(CROP_BENCHMARKS.keys())
    analysis["hardware_connected"] = True
    return jsonify(analysis)

@app.route("/api/export_csv", methods=["GET"])
def api_export_csv():
    hours_param = request.args.get("hours", "24")
    import io, csv
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            if hours_param == "all":
                cursor.execute("""
                    SELECT timestamp, moisture, temp, ec, ph, n, p, k, canopy_pct, exg, yellowing_pct, diagnosis, pump_action 
                    FROM telemetry_logs ORDER BY id ASC
                """)
            else:
                try:
                    h = int(hours_param)
                except ValueError:
                    h = 24
                cursor.execute(f"""
                    SELECT timestamp, moisture, temp, ec, ph, n, p, k, canopy_pct, exg, yellowing_pct, diagnosis, pump_action 
                    FROM telemetry_logs 
                    WHERE timestamp >= datetime('now', '-{h} hours')
                    ORDER BY id ASC
                """)
            rows = cursor.fetchall()
            if not rows:
                cursor.execute("""
                    SELECT timestamp, moisture, temp, ec, ph, n, p, k, canopy_pct, exg, yellowing_pct, diagnosis, pump_action 
                    FROM telemetry_logs ORDER BY id DESC LIMIT 200
                """)
                rows = list(reversed(cursor.fetchall()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Timestamp_UTC", "Moisture_Pct", "Temperature_C", "EC_uS_cm", "Soil_pH",
        "Nitrogen_mg_kg", "Phosphorus_mg_kg", "Potassium_mg_kg",
        "Canopy_Cover_Pct", "Excess_Green_ExG", "Chlorosis_Pct",
        "AI_Multimodal_Diagnosis", "Pump_Actuator_Action"
    ])
    for r in rows:
        writer.writerow(r)

    csv_data = output.getvalue()
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"field_telemetry_{hours_param}h_{timestamp_str}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

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
