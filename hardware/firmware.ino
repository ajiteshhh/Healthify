#include <Arduino.h>
#include <bootanimation.h>
#include <Wire.h>
#include "MAX30100_PulseOximeter.h"
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <WiFi.h>
#include <ArduinoWebsockets.h>
using namespace websockets;

#define PIN_BUTTON 4
#define PIN_ONE_WIRE 14
#define PIN_ECG_ADC 34

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_I2C_ADDR 0x3C

const char* WIFI_SSID = "hi";
const char* WIFI_PASSWORD = "12345678";
const char* WS_SERVER_HOST = "10.100.2.116";
const uint16_t WS_SERVER_PORT = 8000;
const char* WS_PATH_VITALS = "/ws/esp32/vitals";
const char* WS_PATH_ECG = "/ws/esp32/ecg";

const uint32_t BUTTON_DEBOUNCE_MS = 20;
const uint32_t BUTTON_LONGPRESS_MS = 800;

const TickType_t POX_POLL_PERIOD_MS = pdMS_TO_TICKS(10);
const TickType_t VITALS_COMPUTE_PERIOD_MS = pdMS_TO_TICKS(1000);
const TickType_t UI_REFRESH_PERIOD_MS = pdMS_TO_TICKS(50);
const TickType_t ECG_SAMPLE_PERIOD_MS = pdMS_TO_TICKS(3);
const TickType_t CONN_MANAGER_PERIOD_MS = pdMS_TO_TICKS(3000);

Adafruit_SSD1306 g_display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
PulseOximeter g_poxSensor;
OneWire g_oneWire(PIN_ONE_WIRE);
DallasTemperature g_dallasTempSensor(&g_oneWire);

WebsocketsClient g_wsClientVitals;
WebsocketsClient g_wsClientEcg;

SemaphoreHandle_t g_vitalsMutex;
SemaphoreHandle_t g_i2cMutex;

volatile float g_latestBPM = 0.0f;
volatile float g_latestSpO2 = 0.0f;
volatile float g_latestTempC = 0.0f;

volatile float g_latestEcgMV = 0.0f;

volatile bool g_vitalsNeedReconnect = false;
volatile bool g_ecgNeedReconnect = false;

enum UIState { STATE_MENU, STATE_PAGE };
volatile UIState g_uiState = STATE_MENU;
volatile int g_menuIndex = 0;
volatile int g_currentPage = -1;
volatile bool g_isEcgTaskActive = false;

static const unsigned char PROGMEM iconHeart[] = {
  0b00000111,0b11100000, 0b00011111,0b11111000, 0b00111111,0b11111100, 0b01111111,0b11111110,
  0b01111111,0b11111110, 0b01111111,0b11111110, 0b00111111,0b11111100, 0b00011111,0b11111000,
  0b00001111,0b11110000, 0b00000111,0b11100000, 0b00000011,0b11000000, 0b00000001,0b10000000,
  0b00000000,0b00000000, 0b00000000,0b00000000, 0b00000000,0b00000000, 0b00000000,0b00000000,
};
static const unsigned char PROGMEM iconDrop[] = {
  0b00000100,0b00000000, 0b00001110,0b00000000, 0b00011111,0b00000000, 0b00111111,0b10000000,
  0b01111111,0b11000000, 0b11111111,0b11100000, 0b11111111,0b11100000, 0b11111111,0b11100000,
  0b01111111,0b11000000, 0b00111111,0b10000000, 0b00011111,0b00000000, 0b00001110,0b00000000,
  0b00000100,0b00000000, 0b00000000,0b00000000, 0b00000000,0b00000000, 0b00000000,0b00000000,
};
static const unsigned char PROGMEM iconThermo[] = {
  0b00000110,0b00000000, 0b00000110,0b00000000, 0b00000110,0b00000000, 0b00000110,0b00000000,
  0b00000110,0b00000000, 0b00000110,0b00000000, 0b00000110,0b00000000, 0b00001111,0b00000000,
  0b00011111,0b10000000, 0b00011111,0b10000000, 0b00011111,0b10000000, 0b00001111,0b00000000,
  0b00000110,0b00000000, 0b00000000,0b00000000, 0b00000000,0b00000000, 0b00000000,0b00000000,
};
static const unsigned char PROGMEM iconECG[] = {
  0b00000000,0b00000000, 0b00000000,0b00000000, 0b00000111,0b00000000, 0b00001111,0b10000000,
  0b00011111,0b11000000, 0b01111000,0b01110000, 0b11100000,0b00111000, 0b11000000,0b00011100,
  0b10000000,0b00001110, 0b00000000,0b00000110, 0b00000000,0b00000000, 0b00000000,0b00000000,
  0b00000000,0b00000000, 0b00000000,0b00000000, 0b00000000,0b00000000, 0b00000000,0b00000000,
};

void task_PollPoxAndWebsockets(void *pvParameters);
void task_ComputeAndSendVitals(void *pvParameters);
void task_UpdateUI(void *pvParameters);
void task_SampleECG(void *pvParameters);
void task_ConnectionManager(void *pvParameters);

void drawMenu();
void drawPage(int page);
void pollButton();
void playBootAnimation();

void onWebsocketMessage(WebsocketsMessage message);

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("System starting...");

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected. IP: " + WiFi.localIP().toString());

  g_vitalsMutex = xSemaphoreCreateMutex();
  if (g_vitalsMutex == NULL) {
    Serial.println("Vitals Mutex creation failed!");
  }
  g_i2cMutex = xSemaphoreCreateMutex();
  if (g_i2cMutex == NULL) {
    Serial.println("I2C Mutex creation failed!");
  }
  Serial.println("Mutexes created.");

  g_wsClientVitals.onMessage(onWebsocketMessage);
  g_wsClientEcg.onMessage(onWebsocketMessage);

  String urlVitals = String("ws://") + WS_SERVER_HOST + ":" +
                     WS_SERVER_PORT + WS_PATH_VITALS;
  Serial.println("Connecting to WebSocket (vitals): " + urlVitals);
  while (!g_wsClientVitals.connect(urlVitals)) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nVitals WebSocket connected.");


  String urlECG = String("ws://") + WS_SERVER_HOST + ":" +
                  WS_SERVER_PORT + WS_PATH_ECG;
  Serial.println("Connecting to WebSocket (ECG): " + urlECG);
  while (!g_wsClientEcg.connect(urlECG)) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nECG WebSocket connected.");

  Serial.println("Initializing display...");
  if (xSemaphoreTake(g_i2cMutex, portMAX_DELAY) == pdTRUE) {
    if (!g_display.begin(SSD1306_SWITCHCAPVCC, OLED_I2C_ADDR)) {
      Serial.println("SSD1306 allocation failed");
    } else {
      g_display.clearDisplay();
      g_display.setTextSize(1);
      g_display.setTextColor(SSD1306_WHITE);
      g_display.setCursor(0, 0);
      g_display.println("OLED init OK");
      g_display.display();
    }
    xSemaphoreGive(g_i2cMutex);
  }
  delay(1000);

  pinMode(PIN_BUTTON, INPUT_PULLUP);
  g_dallasTempSensor.begin();

  Serial.println("Initializing pulse oximeter...");
  if (xSemaphoreTake(g_i2cMutex, portMAX_DELAY) == pdTRUE) {
    if (!g_poxSensor.begin()) {
      Serial.println("POX init FAILED");
    } else {
      Serial.println("POX init SUCCESS");
      g_poxSensor.setIRLedCurrent(MAX30100_LED_CURR_7_6MA);
    }
    xSemaphoreGive(g_i2cMutex);
  }
  Serial.println("Sensors initialized.");

  xTaskCreatePinnedToCore(task_PollPoxAndWebsockets, "PollTask", 8192, NULL, 3, NULL, 0);
  xTaskCreatePinnedToCore(task_ComputeAndSendVitals, "ComputeTask", 4096, NULL, 2, NULL, 0);
  xTaskCreatePinnedToCore(task_ConnectionManager, "ConnManagerTask", 4096, NULL, 1, NULL, 0);

  playBootAnimation();

  xTaskCreatePinnedToCore(task_UpdateUI, "UITask", 4096, NULL, 3, NULL, 1);
  
  Serial.println("Scheduler started. All tasks running.");
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}

void task_PollPoxAndWebsockets(void *pvParameters) {
  (void) pvParameters;
  TickType_t lastWakeTime = xTaskGetTickCount();

  for (;;) {
    if (g_wsClientVitals.available()) {
      g_wsClientVitals.poll();
    } else {
      g_vitalsNeedReconnect = true;
    }

    if (g_wsClientEcg.available()) {
      g_wsClientEcg.poll();
    } else {
      g_ecgNeedReconnect = true;
    }

    if (xSemaphoreTake(g_i2cMutex, (TickType_t)5) == pdTRUE) {
      g_poxSensor.update();
      xSemaphoreGive(g_i2cMutex);
    }

    vTaskDelayUntil(&lastWakeTime, POX_POLL_PERIOD_MS);
  }
}

void task_ConnectionManager(void *pvParameters) {
  (void) pvParameters;
  TickType_t lastWakeTime = xTaskGetTickCount();

  for (;;) {
    if (g_vitalsNeedReconnect) {
      Serial.println("Manager: Vitals client disconnected. Reconnecting...");
      String urlVitals = String("ws://") + WS_SERVER_HOST + ":" +
                         WS_SERVER_PORT + WS_PATH_VITALS;
      
      if (g_wsClientVitals.connect(urlVitals)) {
        Serial.println("Manager: Vitals reconnected.");
        g_vitalsNeedReconnect = false;
      } else {
        Serial.println("Manager: Vitals reconnect failed, will try again...");
      }
    }

    if (g_ecgNeedReconnect) {
      Serial.println("Manager: ECG client disconnected. Reconnecting...");
      String urlECG = String("ws://") + WS_SERVER_HOST + ":" +
                      WS_SERVER_PORT + WS_PATH_ECG;

      if (g_wsClientEcg.connect(urlECG)) {
        Serial.println("Manager: ECG reconnected.");
        g_ecgNeedReconnect = false;
      } else {
        Serial.println("Manager: ECG reconnect failed, will try again...");
      }
    }

    vTaskDelayUntil(&lastWakeTime, CONN_MANAGER_PERIOD_MS);
  }
}

void task_ComputeAndSendVitals(void *pvParameters) {
  (void) pvParameters;
  TickType_t lastWakeTime = xTaskGetTickCount();

  for (;;) {
    float hr = g_poxSensor.getHeartRate();
    float spo2 = g_poxSensor.getSpO2();
    g_dallasTempSensor.requestTemperatures();
    float tempC = g_dallasTempSensor.getTempCByIndex(0);

    if (xSemaphoreTake(g_vitalsMutex, (TickType_t)10) == pdTRUE) {
      g_latestBPM = hr;
      g_latestSpO2 = spo2;
      g_latestTempC = tempC;
      xSemaphoreGive(g_vitalsMutex);
    }

    String payload = "{\"bpm\": " + String(hr) +
                     ", \"spo2\": " + String(spo2) +
                     ", \"temp\": " + String(tempC) + "}";

    if (g_wsClientVitals.available()) {
      g_wsClientVitals.send(payload);
    }

    vTaskDelayUntil(&lastWakeTime, VITALS_COMPUTE_PERIOD_MS);
  }
}

void task_UpdateUI(void *pvParameters) {
  (void) pvParameters;
  TickType_t lastWakeTime = xTaskGetTickCount();

  for (;;) {
    pollButton();

    if (g_currentPage == 3 && !g_isEcgTaskActive) {
      g_isEcgTaskActive = true;
      Serial.println("UI: Entered ECG page, starting sampling task...");
      xTaskCreatePinnedToCore(task_SampleECG, "ECGSampleTask", 4096, NULL, 2, NULL, 1);
    
    } else if (g_currentPage != 3 && g_isEcgTaskActive) {
      g_isEcgTaskActive = false;
      Serial.println("UI: Left ECG page, stopping sampling task.");
    }

    if (g_uiState == STATE_MENU) {
      drawMenu();
    } else {
      drawPage(g_currentPage);
    }

    vTaskDelayUntil(&lastWakeTime, UI_REFRESH_PERIOD_MS);
  }
}

void task_SampleECG(void *pvParameters) {
  (void) pvParameters;
  TickType_t lastWakeTime = xTaskGetTickCount();

  while (true) {
    if (!g_isEcgTaskActive) {
      Serial.println("ECG Task: g_isEcgTaskActive=false. Deleting self.");
      vTaskDelete(NULL);
    }

    int raw = analogRead(PIN_ECG_ADC);
    float mV = (raw / 4095.0f) * 3300.0f;
    g_latestEcgMV = mV;

    uint8_t data[4];
    memcpy(data, &mV, 4);
    if (g_wsClientEcg.available()) {
      g_wsClientEcg.sendBinary((const char*)data, 4);
    }

    vTaskDelayUntil(&lastWakeTime, ECG_SAMPLE_PERIOD_MS);
  }
}

void playBootAnimation() {
  for (int i = 0; i < totalFrames; i++) {
    if (xSemaphoreTake(g_i2cMutex, portMAX_DELAY) == pdTRUE) {
      g_display.clearDisplay();
      g_display.drawBitmap(0, 0, frames[i], SCREEN_WIDTH, SCREEN_HEIGHT, SSD1306_WHITE);
      g_display.display();
      xSemaphoreGive(g_i2cMutex);
    }
    delay(100);
  }
  delay(300);
}