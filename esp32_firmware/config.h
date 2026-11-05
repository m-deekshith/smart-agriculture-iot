#ifndef CONFIG_H
#define CONFIG_H

// -------------------------------------------------------------
// 1. Mobile Hotspot Configuration (When available)
// -------------------------------------------------------------
#define WIFI_SSID "iQOO"
#define WIFI_PASS "123456789"

// -------------------------------------------------------------
// 2. Direct Offline Edge Access Point (Zero Internet / No Router)
// -------------------------------------------------------------
// Connect phone to this Wi-Fi and open: http://192.168.4.1
#define AP_SSID "VerdantFarm-Edge-Node"
#define AP_PASS "12345678"

// -------------------------------------------------------------
// 3. Live Cloud Server Endpoint (Render.com)
// -------------------------------------------------------------
#define CLOUD_API_URL "https://smart-agriculture-iot-o3ni.onrender.com/api/telemetry_edge"

#endif // CONFIG_H
