#include "credentials.h"
#include "mqtt.h"
#include <NeoPixelBus.h> //Makuna
#include <WiFiUdp.h>
#include <ESP8266WiFi.h>
#include <WiFiClient.h>
#include <ESP8266WebServer.h>
#include <PubSubClient.h> // Nick OLeary
#include <ArduinoJson.h> //Benoit Blanchon
#include <NTPClient.h>
#include <time.h>        // for time() ctime() ...

WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "openwrt.lan", 60*60*2, 60000);

#include "credentials.h"

// Set to the number of LEDs in your LED strip
#define NUM_LEDS 54
// Maximum number of packets to hold in the buffer. Don't change this.
#define BUFFER_LEN 1024
// Toggles FPS output (1 = print FPS over serial, 0 = disable output)
#define PRINT_FPS 1
#define MQTT_FPS 1

//NeoPixelBus settings
const uint8_t PixelPin = 3;  // make sure to set this to the correct pin, ignored for Esp8266(set to 3 by default for DMA)
uint8_t N = 0;

// Wifi and socket settings
unsigned int localPort = 7777;
char packetBuffer[BUFFER_LEN];

const int led = LED_BUILTIN;

// LED strip
NeoPixelBus<NeoGrbFeature, Neo800KbpsMethod> ledstrip(NUM_LEDS, PixelPin);

WiFiUDP port;

int sensorNumber = 1;
unsigned long millisWhenLastDataWasReceived = 0;

#if PRINT_FPS
    uint16_t fpsCounter = 0;
    uint32_t secondTimer = 0;
#endif

WiFiClient wifiClient;
PubSubClient client(wifiClient);

ESP8266WebServer server(80);
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <title>Led Vis</title>
</head>
<body>
    <h1>Led Visualisation</h1>
    <h1><a href='/off'>Breathe Off</a></h1><br>
    <h1><a href='/on'>Breathe On</a></h1>
</body>
</html>
)rawliteral";

void mqttCallback(char* topic, byte* payload, unsigned int length)
{
    payload[length] = '\0';
    String value = String((char*) payload);
    Serial.println(topic);
    Serial.println(value);
    if (value == "OFF") {
      handleBreatheOff();
    }
    if (value == "ON") {
      handleBreatheOn();
    }
}
void reconnectMQTT() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Create a random client ID
    String clientId = "ESP8266Client-";
    clientId += String(random(0xffff), HEX);
    // Attempt to connect
    if (client.connect(clientId.c_str(), MQTT_USERNAME, MQTT_PASSWORD)) {
      Serial.println("connected");
      client.subscribe(MQTT_COMMAND_TOPIC);
      sendMQTTVisLedDiscoveryMsg();
      handleBreatheOn();
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}
void sendMQTTVisLedDiscoveryMsg() {
  DynamicJsonDocument doc(1024);
  char buffer[256];
  doc["name"] = "Visualisation Led";
  doc["command_topic"] = MQTT_COMMAND_TOPIC;
  doc["state_topic"] = MQTT_STATE_TOPIC;
  doc["pl_on"]    = MQTT_COMMAND_ON;
  doc["pl_off"]   = MQTT_COMMAND_OFF;
  doc["stat_t"]   = MQTT_STATE_TOPIC;
  size_t n = serializeJson(doc, buffer);
  client.publish(MQTT_DISCOVERY_TOPIC, buffer, n);
}

void setInternalTime(uint64_t epoch = 0, uint32_t us = 0)
{
  struct timeval tv;
  tv.tv_sec = epoch;
  tv.tv_usec = us;
  settimeofday(&tv, NULL);
}

void setup() {
    pinMode(led, OUTPUT);     // Initialize the LED_BUILTIN pin as an output

    digitalWrite(led, LOW);
    Serial.begin(115200);
    WiFi.hostname(HOSTNAME);
    WiFi.begin(STASSID,STAPSK);
    Serial.println("");
    // Connect to wifi and print the IP address over serial
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.print("Connected to ");
    Serial.println(STASSID);
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    Serial.println(WiFi.macAddress());
    
    digitalWrite(led, HIGH);
    delay(500);
    digitalWrite(led, LOW);
    port.begin(localPort);
    ledstrip.Begin();//Begin output
    ledstrip.Show();//Clear the strip for use
    digitalWrite(led, HIGH);

    server.on("/",handleRoot);
    server.on("/off",handleBreatheOff);
    server.on("/on",handleBreatheOn);
    server.begin();

    client.setServer(MQTT_SERVER, MQTT_PORT);
    client.setCallback(mqttCallback);

    timeClient.begin();
    timeClient.update();
    Serial.print("Time: ");
    Serial.println(timeClient.getFormattedTime());
    Serial.println(timeClient.getEpochTime());
    setTime(timeClient.getEpochTime());
}

boolean breathe = true;
int MIN_X = -360;
float MIN_Y = abs(atan(MIN_X/40));
int x = MIN_X;
uint8_t fadeDirection = 1;
int a=32;
int c=50;

float nextY() {
  float y = floor(a*exp(-pow(x,2)/(2*pow(c,2))));
  //Serial.print("Fading ");  Serial.print(x);  Serial.print(" ");  Serial.println(y);
  x += fadeDirection;
  delay(5);
  if (x >= -MIN_X) {
    x = MIN_X;
    delay(1000);
  }
  return y;
}

void sendMQTTState() {
  DynamicJsonDocument doc(1024);
  char buffer[256];
  //doc["breathe"] = breathe;
  size_t n = serializeJson(doc, buffer);
  if (breathe) {
    client.publish(MQTT_STATE_TOPIC, MQTT_COMMAND_ON);
  } else {
    client.publish(MQTT_STATE_TOPIC, MQTT_COMMAND_OFF);
  }
  #if MQTT_FPS
    char *fpss = itoa(fpsCounter,buffer,10);
    Serial.print("MQTT FPS: ");
    Serial.println(fpss);
    client.publish(MQTT_FPS_TOPIC, fpss);
  #endif
}

void loop() {
  if (!client.connected()) {
    reconnectMQTT();
  }
  client.loop();
  // Read data over socket
  int packetSize = port.parsePacket();
  // If packets have been received, interpret the command
  if (packetSize) {
      millisWhenLastDataWasReceived = millis();
      int len = port.read(packetBuffer, BUFFER_LEN);
      // Serial.print("Received ");
      // Serial.print(len);
      // Serial.println(" bytes");
      for(int i = 0; i < len; i+=4) {
          digitalWrite(led, LOW);
          packetBuffer[len] = 0;
          N = packetBuffer[i];
          RgbColor pixel((uint8_t)packetBuffer[i+1], (uint8_t)packetBuffer[i+2], (uint8_t)packetBuffer[i+3]);
          ledstrip.SetPixelColor(N, pixel);
          digitalWrite(led, HIGH);
      } 
      ledstrip.Show();
      #if PRINT_FPS
          fpsCounter++;
          //Serial.print("/");//Monitors connection(shows jumps/jitters in packets)
      #endif
    } else if (breathe) {
      if (millis() - millisWhenLastDataWasReceived > 3000) {
        float fadeY = nextY();
        
        for(int i = 0; i < NUM_LEDS; i+=1) {
            digitalWrite(led, LOW);
            RgbColor pixel(fadeY, 0, fadeY);
            ledstrip.SetPixelColor(i, pixel);
            digitalWrite(led, HIGH);
        }
        ledstrip.Show();
      }
    }
    #if PRINT_FPS
      if (millis() - secondTimer >= 1000U) {
        secondTimer = millis();
        Serial.printf("FPS: %d\n", fpsCounter);
        #if MQTT_FPS
          sendMQTTState();
        #endif
        fpsCounter = 0;
      }
    #endif

    server.handleClient();
}

void handleRoot() {
  server.send(200,"text/html",index_html);  
}
void handleBreatheOff() {
  breathe = false;
  for(int i = 0; i < NUM_LEDS; i+=1) {
      digitalWrite(led, LOW);
      RgbColor pixel(0, 0, 0);
      ledstrip.SetPixelColor(i, pixel);
      digitalWrite(led, HIGH);
  }
  ledstrip.Show();
  sendMQTTState();
  server.send(200,"text/html",index_html);
}
void handleBreatheOn() {
  breathe = true;
  sendMQTTState();
  server.send(200,"text/html",index_html);
}
