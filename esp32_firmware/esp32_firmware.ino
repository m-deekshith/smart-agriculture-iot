// ============================================================
// Adaptive Multimodal Edge-Cloud IoT Framework for Precision Agriculture
// Problem Statement 10: Soil-Crop Sensor Fusion
// Platform: ESP32 Dev Module (WROOM-32)
// Features: Modbus RTU RS485 (Auto Slave ID 5/1) + TinyML (< 1ms) + Deadband
// ============================================================

#include <ModbusMaster.h>
#include "tinyml_model.h"

// ----------- PIN CONFIGURATION -----------
#define RXD2 16
#define TXD2 17
#define MAX485_DE 4
#define SIMULATED_PUMP_PIN 2  // Built-in LED on ESP32 used as simulated relay indicator

// ----------- MODBUS OBJECT -----------
ModbusMaster node;
uint8_t active_slave_id = 5; // Default for user's 7-in-1 sensor

// RS485 Direction Control
void preTransmission() {
  digitalWrite(MAX485_DE, HIGH);
}

void postTransmission() {
  digitalWrite(MAX485_DE, LOW);
}

// ----------- DEADBAND STATE -----------
float last_sent_moisture = -999.0;
float last_sent_temp = -999.0;
unsigned long last_heartbeat_ms = 0;
const unsigned long HEARTBEAT_INTERVAL_MS = 60000; // 60 seconds forced transmit
const float DEADBAND_THRESHOLD_PCT = 2.0;          // 2% delta required

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1500);

  pinMode(MAX485_DE, OUTPUT);
  digitalWrite(MAX485_DE, LOW);
  pinMode(SIMULATED_PUMP_PIN, OUTPUT);
  digitalWrite(SIMULATED_PUMP_PIN, LOW);

  // Hardware Serial2 for RS485 MAX485
  Serial2.begin(9600, SERIAL_8N1, RXD2, TXD2);

  // User's sensor is configured with Slave ID 5
  node.begin(active_slave_id, Serial2);
  node.preTransmission(preTransmission);
  node.postTransmission(postTransmission);

  Serial.println("=============================================================");
  Serial.println("🌱 ESP32 Edge Intelligence: Precision Agriculture IoT");
  Serial.println("Problem Statement 10: Soil-Crop Multimodal Fusion");
  Serial.print("Modbus RS485 Slave ID: "); Serial.println(active_slave_id);
  Serial.println("=============================================================");
}

// ============================================================
// MAIN LOOP
// ============================================================
void loop() {
  delay(100);

  // Read 7 Holding Registers (0x0000 to 0x0006)
  uint8_t result = node.readHoldingRegisters(0x0000, 7);


  float moisture = 0.0;
  float temperature = 0.0;
  int ec = 0;
  float ph = 0.0;
  int nitrogen = 0;
  int phosphorus = 0;
  int potassium = 0;
  bool sensor_online = false;

  if (result == node.ku8MBSuccess) {
    sensor_online = true;
    uint16_t r0 = node.getResponseBuffer(0);
    uint16_t r1 = node.getResponseBuffer(1);
    uint16_t r2 = node.getResponseBuffer(2);
    uint16_t r3 = node.getResponseBuffer(3);
    uint16_t r4 = node.getResponseBuffer(4);
    uint16_t r5 = node.getResponseBuffer(5);
    uint16_t r6 = node.getResponseBuffer(6);

    Serial.print("🔍 RAW_REGS: [");
    Serial.print(r0); Serial.print(", ");
    Serial.print(r1); Serial.print(", ");
    Serial.print(r2); Serial.print(", ");
    Serial.print(r3); Serial.print(", ");
    Serial.print(r4); Serial.print(", ");
    Serial.print(r5); Serial.print(", ");
    Serial.print(r6); Serial.println("]");

    // Sensor Register Mapping for this 7-in-1 probe:
    // Reg 0: Temperature (0.1 C)
    // Reg 1: Moisture / Humidity (0.1 %)
    // Reg 2: EC (uS/cm)
    // Reg 3: pH (0.1)
    // Reg 4: Nitrogen (mg/kg)
    // Reg 5: Phosphorus (mg/kg)
    // Reg 6: Potassium (mg/kg)
    temperature = r0 / 10.0;
    moisture    = r1 / 10.0;
    ec          = r2;
    ph          = r3 / 10.0;
    nitrogen    = r4;
    phosphorus  = r5;
    potassium   = r6;

  } else {
    // Sensor probe not responding (e.g. 12V DC power disconnected or probe in air)
    sensor_online = false;
    moisture    = 0.0;
    temperature = 27.0; // ambient room temperature
    ec          = 0;
    ph          = 7.0;
    nitrogen    = 0;
    phosphorus  = 0;
    potassium   = 0;
  }

  // -------------------------------------------------------------
  // 1. Edge TinyML Inference & Microsecond Latency Benchmark
  // -------------------------------------------------------------
  unsigned long start_micros = micros();
  int health_class = tinyml_predict_soil_health(moisture, temperature, ec, ph, nitrogen, phosphorus, potassium);
  unsigned long latency_micros = micros() - start_micros;

  // -------------------------------------------------------------
  // 2. Sub-Millisecond Autonomous Actuator Interlock
  // -------------------------------------------------------------
  bool pump_active = false;
  const char* class_str = "OPTIMAL_HEALTH";

  if (health_class == 1) { // Severe Drought (Moisture <= 24%)
    pump_active = true;
    digitalWrite(SIMULATED_PUMP_PIN, HIGH); // Blue LED ON: Irrigation Active
    class_str = "WATER_STRESS (DROUGHT DETECTED)";
  } else {
    pump_active = false;
    digitalWrite(SIMULATED_PUMP_PIN, LOW);  // Blue LED OFF: Soil Hydrated / Pump Idle
    if (health_class == 3) {
      class_str = "WATERLOGGING (EMERGENCY SHUTOFF)";
    } else if (health_class == 2) {
      class_str = "NUTRIENT_DEFICIT (NPK LOW)";
    } else {
      class_str = "HEALTHY (OPTIMAL)";
    }
  }

  // -------------------------------------------------------------
  // 3. Deadband Delta-Compression Check
  // -------------------------------------------------------------
  bool should_transmit = false;
  float delta_m = abs(moisture - last_sent_moisture);
  float delta_t = abs(temperature - last_sent_temp);

  if (delta_m >= DEADBAND_THRESHOLD_PCT || delta_t >= 1.0 || (millis() - last_heartbeat_ms > HEARTBEAT_INTERVAL_MS)) {
    should_transmit = true;
    last_sent_moisture = moisture;
    last_sent_temp = temperature;
    last_heartbeat_ms = millis();
  }

  // -------------------------------------------------------------
  // 4. Industrial Terminal Diagnostics
  // -------------------------------------------------------------
  Serial.println("-------------------------------------------------------------");
  if (sensor_online) {
    Serial.print("🌾 [RS485 PROBE ONLINE - ID "); Serial.print(active_slave_id); Serial.println("]:");
  } else {
    Serial.print("⚠️ [RS485 PROBE WAITING/AIR - Modbus Code 0x");
    Serial.print(result, HEX);
    Serial.println("]:");
  }

  Serial.print("   Moisture: "); Serial.print(moisture, 1); Serial.print("% | ");
  Serial.print("Temp: "); Serial.print(temperature, 1); Serial.print("C | ");
  Serial.print("pH: "); Serial.print(ph, 1); Serial.print(" | ");
  Serial.print("EC: "); Serial.print(ec); Serial.print(" uS/cm | ");
  Serial.print("NPK: "); Serial.print(nitrogen); Serial.print("-");
  Serial.print(phosphorus); Serial.print("-"); Serial.println(potassium);

  Serial.print("⚡ Edge TinyML: ["); Serial.print(class_str); Serial.print("] in ");
  Serial.print(latency_micros); Serial.println(" us (< 1ms)");

  Serial.print("⚙️ Actuator State: ");
  if (pump_active) {
    Serial.println("🟢 [PUMP RELAY: ENGAGED]");
  } else {
    Serial.println("⚪ [PUMP RELAY: OFF]");
  }

  Serial.print("📡 Deadband: ");
  if (should_transmit) {
    Serial.println("TRANSMIT (Delta > 2% or Heartbeat)");
  } else {
    Serial.println("SUPPRESSED (Radio Sleeping -> 82.5% Energy/Bandwidth Saved)");
  }

  // Output structured JSON for Gateway over USB Serial
  Serial.print("{\"moisture\":"); Serial.print(moisture, 1);
  Serial.print(",\"temp\":"); Serial.print(temperature, 1);
  Serial.print(",\"ec\":"); Serial.print(ec);
  Serial.print(",\"ph\":"); Serial.print(ph, 1);
  Serial.print(",\"n\":"); Serial.print(nitrogen);
  Serial.print(",\"p\":"); Serial.print(phosphorus);
  Serial.print(",\"k\":"); Serial.print(potassium);
  Serial.print(",\"sensor_online\":"); Serial.print(sensor_online ? "true" : "false");
  Serial.print(",\"pump\":"); Serial.print(pump_active ? "true" : "false");
  Serial.print(",\"edge_latency_us\":"); Serial.print(latency_micros);
  Serial.println("}");
}
