// Workshop Scribe — MCU side (Arduino Uno Q, STM32U585)
//
// This sketch handles the "body" half of the dual-brain. It:
//   - Provides a visible status indicator on the 8x13 LED matrix
//     ("idle", "watching", "noticed something", "thinking", "done")
//   - Reads a non-invasive AC current sensor (SCT-013 on A0) clipped
//     around the soldering iron's mains lead, so we know when the
//     iron is on without touching mains. Notifies the MPU on edges.
//   - Reads a momentary push-button on D2 — short press = "mark this
//     moment", long press (>2s) = "end session"
//   - Pulses the RGB LEDs as a glanceable activity indicator
//
// The MPU calls into us via Bridge.provide("set_status", ...) etc.
// We notify the MPU of hardware events via Bridge.notify("...").
//
// Notes on libraries: this targets the Arduino UNO Q under Zephyr
// (FQBN arduino:zephyr:unoq). Bridge comes from the App Lab template;
// Arduino_LED_Matrix is the standard library for the onboard matrix.

#include <Arduino.h>
#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>  // Bridge / RPC for App Lab

// ---------- Pins ----------
constexpr uint8_t PIN_BUTTON      = 2;   // momentary, INPUT_PULLUP
constexpr uint8_t PIN_IRON_SENSOR = A0;  // SCT-013 current clamp via burden resistor
constexpr uint8_t PIN_RGB_R       = 6;   // user RGB LEDs (MCU-controllable)
constexpr uint8_t PIN_RGB_G       = 5;
constexpr uint8_t PIN_RGB_B       = 3;

// ---------- Status states (kept in sync with main.py) ----------
enum Status : uint8_t {
  STATUS_IDLE     = 0,
  STATUS_WATCHING = 1,
  STATUS_NOTICED  = 2,  // transient — auto-decays back to WATCHING
  STATUS_THINKING = 3,  // end-of-session: Opus call in flight
  STATUS_DONE     = 4,
};

// ---------- Globals ----------
ArduinoLEDMatrix matrix;
Status currentStatus = STATUS_IDLE;
unsigned long lastFrameMs = 0;
unsigned long noticedSinceMs = 0;
constexpr unsigned long FRAME_INTERVAL_MS = 80;     // ~12 fps animation
constexpr unsigned long NOTICED_DURATION_MS = 1500; // pulse duration

// Soldering iron detection (debounced edge-trigger).
// SCT-013 with a 10R burden gives a few hundred mV peak at typical iron draw.
// We sample a window, take peak-to-peak, and threshold.
constexpr int IRON_SAMPLES = 200;
constexpr int IRON_THRESHOLD_PP = 60;   // ADC counts; tune on bench
constexpr unsigned long IRON_POLL_MS = 500;
unsigned long lastIronPollMs = 0;
bool ironOn = false;

// Button (debounced, distinguishes short vs long press).
unsigned long buttonDownMs = 0;
bool buttonHeld = false;
constexpr unsigned long DEBOUNCE_MS = 30;
constexpr unsigned long LONG_PRESS_MS = 2000;

// Animation state.
uint8_t frameBuf[8][13];
uint8_t pulsePhase = 0;

// ---------- Forward decls ----------
void rpc_set_status(int s);
void rpc_pulse_led(String kind);
void renderFrame();
void pollIronSensor();
void pollButton();
bool readIronOn();

// ---------- Setup ----------
void setup() {
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_RGB_R, OUTPUT);
  pinMode(PIN_RGB_G, OUTPUT);
  pinMode(PIN_RGB_B, OUTPUT);
  analogReadResolution(12);   // STM32U585 supports 12-bit ADC

  matrix.begin();

  Bridge.begin();
  // Expose two functions the MPU can call.
  Bridge.provide("set_status", rpc_set_status);
  Bridge.provide("pulse_led", rpc_pulse_led);
}

// ---------- Loop ----------
void loop() {
  unsigned long now = millis();

  // 1. Auto-decay the "noticed" pulse back to watching.
  if (currentStatus == STATUS_NOTICED &&
      (now - noticedSinceMs) > NOTICED_DURATION_MS) {
    currentStatus = STATUS_WATCHING;
  }

  // 2. Animation tick.
  if (now - lastFrameMs >= FRAME_INTERVAL_MS) {
    lastFrameMs = now;
    pulsePhase++;
    renderFrame();
  }

  // 3. Sensors.
  if (now - lastIronPollMs >= IRON_POLL_MS) {
    lastIronPollMs = now;
    pollIronSensor();
  }
  pollButton();
}

// ---------- RPC handlers (called by MPU) ----------

// Called by the Python agent when its state changes.
void rpc_set_status(int s) {
  if (s < 0 || s > STATUS_DONE) return;
  currentStatus = static_cast<Status>(s);
}

// Called by the Python agent when it logs an event of interest.
// We flash to NOTICED briefly so it's visible on camera.
void rpc_pulse_led(String kind) {
  (void)kind;
  if (currentStatus == STATUS_WATCHING || currentStatus == STATUS_NOTICED) {
    currentStatus = STATUS_NOTICED;
    noticedSinceMs = millis();
  }
}

// ---------- Sensor polling ----------

bool readIronOn() {
  int minV = 4095, maxV = 0;
  for (int i = 0; i < IRON_SAMPLES; i++) {
    int v = analogRead(PIN_IRON_SENSOR);
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }
  return (maxV - minV) > IRON_THRESHOLD_PP;
}

void pollIronSensor() {
  bool nowOn = readIronOn();
  if (nowOn != ironOn) {
    ironOn = nowOn;
    // Fire-and-forget notification to the MPU. Matches the
    // bridge.subscribe("mcu_events") loop in main.py.
    Bridge.call("mcu_events", String(ironOn ? "iron_on" : "iron_off"));
  }
}

void pollButton() {
  static unsigned long lastChangeMs = 0;
  static int lastReading = HIGH;

  int reading = digitalRead(PIN_BUTTON);
  unsigned long now = millis();

  if (reading != lastReading) {
    lastChangeMs = now;
    lastReading = reading;
  }

  if ((now - lastChangeMs) < DEBOUNCE_MS) return;

  // Stable state. Pressed = LOW (pull-up).
  if (reading == LOW && buttonDownMs == 0) {
    buttonDownMs = now;
    buttonHeld = false;
  }
  if (reading == LOW && !buttonHeld &&
      buttonDownMs && (now - buttonDownMs) >= LONG_PRESS_MS) {
    buttonHeld = true;
    Bridge.call("mcu_events", String("session_end"));
  }
  if (reading == HIGH && buttonDownMs != 0) {
    if (!buttonHeld) {
      // Short press = mark moment.
      Bridge.call("mcu_events", String("marker_pressed"));
    }
    buttonDownMs = 0;
    buttonHeld = false;
  }
}

// ---------- Rendering ----------
//
// Each status has a distinct visual personality so the camera can
// pick it up at a glance:
//   IDLE      — single dim dot, top-left, slow heartbeat
//   WATCHING  — two dots gently scanning left-to-right
//   NOTICED   — full matrix flash that fades over ~1.5s
//   THINKING  — diagonal sweep, like it's mulling something over
//   DONE      — checkmark glyph, solid

void clearFrame() {
  for (int y = 0; y < 8; y++)
    for (int x = 0; x < 13; x++)
      frameBuf[y][x] = 0;
}

void renderIdle() {
  clearFrame();
  uint8_t b = (sin(pulsePhase * 0.05) + 1.0) * 30;  // very dim heartbeat
  frameBuf[0][0] = b;
}

void renderWatching() {
  clearFrame();
  // Two dots scanning across row 4 in opposite directions.
  uint8_t a = pulsePhase % 26;
  uint8_t x1 = (a < 13) ? a : (25 - a);
  uint8_t x2 = 12 - x1;
  frameBuf[3][x1] = 80;
  frameBuf[4][x2] = 80;
}

void renderNoticed() {
  // Bright flash that decays linearly over NOTICED_DURATION_MS.
  unsigned long elapsed = millis() - noticedSinceMs;
  uint8_t brightness = elapsed >= NOTICED_DURATION_MS
      ? 0
      : 255 - (elapsed * 255 / NOTICED_DURATION_MS);
  for (int y = 0; y < 8; y++)
    for (int x = 0; x < 13; x++)
      frameBuf[y][x] = brightness;
}

void renderThinking() {
  clearFrame();
  // Diagonal sweep: lit cells where (x + y + phase) is in a band.
  for (int y = 0; y < 8; y++) {
    for (int x = 0; x < 13; x++) {
      uint8_t v = (x + y + pulsePhase) % 16;
      if (v < 4) frameBuf[y][x] = 60 + v * 40;
    }
  }
}

void renderDone() {
  clearFrame();
  // A small checkmark, statically lit.
  static const uint8_t check[][2] = {
    {6, 2}, {5, 3}, {4, 4}, {3, 5}, {2, 6}, {1, 7}, {0, 8},
    {1, 5}, {2, 4}                          // the short stroke
  };
  for (auto& p : check) frameBuf[p[0]][p[1]] = 200;
}

void updateRgbForStatus() {
  switch (currentStatus) {
    // Idle: solid blue. Bench is "ready to record".
    case STATUS_IDLE:
      analogWrite(PIN_RGB_R, 0);   analogWrite(PIN_RGB_G, 0);   analogWrite(PIN_RGB_B, 80);
      break;

    // Watching = recording in progress. Bright red, visible from across
    // the bench so it's obvious a session is live.
    case STATUS_WATCHING:
      analogWrite(PIN_RGB_R, 200); analogWrite(PIN_RGB_G, 0);   analogWrite(PIN_RGB_B, 0);
      break;

    // Noticed: transient amber when something's just been logged.
    case STATUS_NOTICED:
      analogWrite(PIN_RGB_R, 180); analogWrite(PIN_RGB_G, 100); analogWrite(PIN_RGB_B, 0);
      break;

    // Thinking: purple pulse via a triangle wave on millis(). One full
    // cycle per second matches the MPU-side LED1+LED2 toggle so all four
    // LEDs breathe in sync during the Opus call.
    case STATUS_THINKING: {
      uint16_t pos = millis() % 1000;
      uint8_t  tri = pos < 500 ? (uint8_t)(pos / 4) : (uint8_t)((1000 - pos) / 4);  // 0..125..0
      analogWrite(PIN_RGB_R, tri);
      analogWrite(PIN_RGB_G, 0);
      analogWrite(PIN_RGB_B, tri);
    } break;

    // Done: brief green confirmation before falling back to idle.
    case STATUS_DONE:
      analogWrite(PIN_RGB_R, 0);   analogWrite(PIN_RGB_G, 150); analogWrite(PIN_RGB_B, 0);
      break;
  }
}

void renderFrame() {
  switch (currentStatus) {
    case STATUS_IDLE:     renderIdle();     break;
    case STATUS_WATCHING: renderWatching(); break;
    case STATUS_NOTICED:  renderNoticed();  break;
    case STATUS_THINKING: renderThinking(); break;
    case STATUS_DONE:     renderDone();     break;
  }
  matrix.renderBitmap(frameBuf, 8, 13);
  updateRgbForStatus();
}
