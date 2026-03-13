/*
 * AuraTrack Privacy Node (ESP32 #2)
 * ==================================
 * Press button → toggles privacy mode ON/OFF
 * Sends UDP packet to bridge on port 5007:
 *   {"privacy": true}  or  {"privacy": false}
 * 
 * Wiring:
 *   Button → GPIO14 (other leg to GND)
 *   Built-in LED → GPIO2 (ON when privacy active)
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>

const char* SSID       = "tnet";
const char* PASSWORD   = "moonwalk";
const char* SERVER_IP  = "10.131.166.65";
const int   SERVER_PORT = 5007;       // privacy channel

#define BUTTON_PIN  14
#define LED_PIN      2                // built-in LED

WiFiUDP udp;
bool privacyMode      = false;
bool lastButtonState  = HIGH;
uint32_t lastSendTime = 0;

void sendPrivacyState() {
  StaticJsonDocument<64> doc;
  doc["privacy"]   = privacyMode;
  doc["source"]    = "privacy_node";
  doc["timestamp"] = millis();

  char buf[64];
  serializeJson(doc, buf);

  udp.beginPacket(SERVER_IP, SERVER_PORT);
  udp.print(buf);
  udp.endPacket();

  Serial.printf("[UDP] Sent: %s\n", buf);
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASSWORD);
  Serial.print("[WiFi] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());

  udp.begin(4211);
  Serial.printf("[UDP] Sending to %s:%d\n", SERVER_IP, SERVER_PORT);
  Serial.println("[Privacy Node] Ready — press button to toggle");

  // Send initial state
  sendPrivacyState();
}

void loop() {
  bool currentButtonState = digitalRead(BUTTON_PIN);

  // Detect button press (HIGH→LOW)
  if (lastButtonState == HIGH && currentButtonState == LOW) {
    privacyMode = !privacyMode;
    digitalWrite(LED_PIN, privacyMode ? HIGH : LOW);

    Serial.println(privacyMode ? "🔒 PRIVACY MODE ON" : "🔓 PRIVACY MODE OFF");

    sendPrivacyState();
    delay(300);  // debounce
  }
  lastButtonState = currentButtonState;

  // Re-send state every 2 seconds (so bridge recovers if it restarts)
  if (millis() - lastSendTime >= 2000) {
    lastSendTime = millis();
    sendPrivacyState();
  }
}
