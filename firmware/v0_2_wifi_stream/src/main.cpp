// Sanad firmware v0.2: sensors streamed over Wi-Fi (WebSocket) and USB.
//
// Same sensors and wiring as v0.1. New: every sample also goes over Wi-Fi to the laptop visualizer
// (ws://<band>/ws), and samples come from the MPU's FIFO with a counter `seq` (gaps = lost samples).
//
// Serial commands (type a letter in the monitor; a WebSocket client can send the same letters):
//   h  human-readable summary, 2 lines per second (default)
//   r  raw CSV stream at 50 Hz on USB
//   s  rescan the I2C bus and re-initialise both sensors
//   w  Wi-Fi status: mode, network, IP address, visualizers connected, dropped lines
//   a  switch to the band's own network "Sanad-Band" now
//   j  look for the saved networks again
//
// Stream line, identical on USB (raw mode) and in every WebSocket text message:
//   t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,ir,red,seq
// Lines starting with '#' are comments. Axes are the MPU's chip axes, not yet the Sanad frame.
//
// LED: slow blink = running, no visualizer connected; on with a short blink-off each second =
// streaming to at least one visualizer.

#include <Arduino.h>

#include "config.h"
#include "net.h"
#include "sensors.h"

static bool rawMode = false;

// ---------- line formatting (integers only: this core's printf prints %f as nothing) ----------
static char *putUint(char *p, uint32_t v) {
  char tmp[10];
  int n = 0;
  do {
    tmp[n++] = (char)('0' + v % 10);
    v /= 10;
  } while (v);
  while (n) *p++ = tmp[--n];
  return p;
}

// counts * num / den with `pow10` = 10^decimals, rounded half away from zero, e.g. "-0.023".
static char *putScaled(char *p, int16_t counts, int32_t num, int32_t den, uint32_t pow10) {
  int32_t x = (int32_t)counts * num;
  bool neg = x < 0;
  uint32_t a = neg ? (uint32_t)(-x) : (uint32_t)x;
  uint32_t q = (2 * a + (uint32_t)den) / (2 * (uint32_t)den);
  if (neg && q) *p++ = '-';
  p = putUint(p, q / pow10);
  *p++ = '.';
  uint32_t frac = q % pow10;
  for (uint32_t d = pow10 / 10; d; d /= 10) *p++ = (char)('0' + (frac / d) % 10);
  return p;
}

// "t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,ir,red,seq" without newline; returns its length.
static size_t formatLine(const Sample &s, char *buf) {
  char *p = putUint(buf, s.t_ms);
  for (int k = 0; k < 3; k++) {
    *p++ = ',';
    p = putScaled(p, s.raw[k], 1000, 2048, 1000);  // g, 3 decimals (2048 counts/g)
  }
  for (int k = 3; k < 6; k++) {
    *p++ = ',';
    p = putScaled(p, s.raw[k], 100, 164, 10);  // °/s, 1 decimal (16.4 counts per °/s)
  }
  *p++ = ',';
  p = putUint(p, maxIr);
  *p++ = ',';
  p = putUint(p, maxRed);
  *p++ = ',';
  p = putUint(p, s.seq);
  *p = 0;
  return (size_t)(p - buf);
}

// ---------- human view (as v0.1, plus a Wi-Fi line every 5 s) ----------
static void printNum(const char *label, float v, int digits) {
  Serial.print(label);
  if (v >= 0) Serial.print('+');
  Serial.print(v, digits);
}

static void printHuman(const Sample &s, bool haveSample) {
  static uint32_t last = 0;
  static uint8_t lines = 0;
  if (millis() - last < 500) return;
  last = millis();
  if (mpuOk && haveSample) {
    float ax = accelG(s.raw[0]), ay = accelG(s.raw[1]), az = accelG(s.raw[2]);
    printNum("acc[g] x=", ax, 2);
    printNum(" y=", ay, 2);
    printNum(" z=", az, 2);
    printNum(" |a|=", sqrtf(ax * ax + ay * ay + az * az), 2);
    printNum("  gyro[dps] x=", gyroDps(s.raw[3]), 1);
    printNum(" y=", gyroDps(s.raw[4]), 1);
    printNum(" z=", gyroDps(s.raw[5]), 1);
    printNum("  ", mpuTempC(), 1);
    Serial.print("C   ");
  } else {
    Serial.print("MPU-6050 MISSING   ");
  }
  if (maxOk) {
    Serial.printf("IR=%6lu red=%6lu finger=%s\n", (unsigned long)maxIr, (unsigned long)maxRed,
                  maxIr > FINGER_IR ? "YES" : "no");
  } else {
    Serial.println("MAX30102 MISSING");
  }
  if (lines++ % 10 == 0) {
    char buf[200];
    netSummary(buf, sizeof(buf));
    Serial.println(buf);
  }
}

// ---------- commands (USB and WebSocket) ----------
static void printHelp() {
  logMsg("Sanad v" SANAD_VERSION " Wi-Fi stream. Commands: h = human view, r = raw 50 Hz CSV, "
         "s = rescan sensors, w = Wi-Fi status, a = own network now, j = join saved network again");
}

static void handleCommand(char c) {
  switch (c) {
    case 'h':
      rawMode = false;
      Serial.println("# human mode");
      break;
    case 'r':
      rawMode = true;
      Serial.println(CSV_HEADER);
      break;
    case 's':
      sensorsStart();
      break;
    case 'w':
      netPrintStatus();
      break;
    case 'a':
      netForceAp();
      break;
    case 'j':
      netRejoin();
      break;
    case '?':
      printHelp();
      break;
    default:
      break;  // newlines etc.
  }
}

// Slow blink = no visualizer; mostly on with a short off-blink each second = streaming.
static void updateLed() {
  static int shown = -1;
  uint32_t t = millis() % 1000;
  int on = netClientCount() > 0 ? (t >= 100) : (t < 500);
  if (on != shown) {
    digitalWrite(LED_BUILTIN, on ? HIGH : LOW);
    shown = on;
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  while (!Serial && millis() < 3000) delay(10);  // give the USB monitor a moment to attach

  printHelp();
  sensorsBegin();
  netBegin(handleCommand);  // a few seconds: scan + join a saved network, or start own network
  sensorsRestartStream();   // seq starts at 0 now that the loop can keep up
}

void loop() {
  while (Serial.available()) handleCommand((char)Serial.read());
  netLoop();
  sensorsRetry();

  static Sample batch[MAX_BATCH];
  static Sample latest;
  static bool haveLatest = false;
  int n = sensorsPoll(batch, MAX_BATCH);
  for (int i = 0; i < n; i++) {
    char line[100];
    size_t len = formatLine(batch[i], line);  // built once, sent to every output
    netSendLine(line, len);
    if (rawMode) {
      line[len] = '\n';
      Serial.write(line, len + 1);
    }
  }
  if (n > 0) {
    latest = batch[n - 1];
    haveLatest = true;
  }

  if (!rawMode) printHuman(latest, haveLatest);
  updateLed();
}
