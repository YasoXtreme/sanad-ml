// Sanad firmware v0.1: sensor wiring test.
//
// Reads the MPU-6050 (accel ±16 g, gyro ±2000 °/s, 50 Hz) and the MAX30102 (red + IR, 50 Hz)
// over I2C0 (SDA = GP4, SCL = GP5) and prints the values over USB serial. No libraries,
// just register reads, so every failure points at wiring and not at a library.
//
// Serial commands (type a letter in the monitor):
//   h  human-readable summary, 2 lines per second (default)
//   r  raw CSV stream at 50 Hz (for a laptop visualizer later)
//   s  rescan the I2C bus and re-initialise both sensors
//
// Axes are the MPU-6050's own chip axes, not yet rotated into the Sanad frame.

#include <Arduino.h>
#include <Wire.h>

constexpr uint8_t MPU_ADDR = 0x68;  // AD0 low (default on most modules)
constexpr uint8_t MAX_ADDR = 0x57;
constexpr uint32_t SAMPLE_US = 20000;  // 50 Hz
constexpr uint32_t FINGER_IR = 50000;  // IR above this = something is on the sensor

bool mpuOk = false;
bool maxOk = false;
bool rawMode = false;
uint8_t mpuWho = 0x68;  // WHO_AM_I: 0x68 = genuine MPU-6050, others = MPU-6500-type clones

float ax, ay, az, gx, gy, gz, tempC;  // g, °/s, °C
uint32_t red = 0, ir = 0;              // 18-bit MAX30102 counts
uint8_t maxNew = 0;                    // new MAX30102 samples in the last read

// ---------- I2C helpers ----------
bool writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool readRegs(uint8_t addr, uint8_t reg, uint8_t *buf, size_t n) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, n) != n) return false;
  for (size_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

void i2cScan() {
  Serial.print("# I2C scan:");
  int found = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.printf(" 0x%02X", a);
      found++;
    }
  }
  if (found == 0) Serial.print(" NOTHING FOUND -> check 3V3, GND, SDA=GP4, SCL=GP5");
  Serial.println("   (expected: 0x57 = MAX30102, 0x68 = MPU-6050)");
}

// ---------- MPU-6050 ----------
bool initMpu() {
  uint8_t who;
  if (!readRegs(MPU_ADDR, 0x75, &who, 1)) {
    Serial.println("# MPU-6050: no answer at 0x68");
    return false;
  }
  mpuWho = who;
  Serial.printf("# MPU-6050 WHO_AM_I = 0x%02X %s\n", who,
                who == 0x68 ? "(ok)" : "(not 0x68: clone or other chip, may still work)");
  bool ok = writeReg(MPU_ADDR, 0x6B, 0x80);  // reset
  delay(100);
  ok &= writeReg(MPU_ADDR, 0x6B, 0x01);  // wake up, clock = gyro X PLL
  ok &= writeReg(MPU_ADDR, 0x1A, 0x03);  // low-pass filter ~44 Hz
  ok &= writeReg(MPU_ADDR, 0x19, 19);    // 1 kHz / (1 + 19) = 50 Hz
  ok &= writeReg(MPU_ADDR, 0x1B, 0x18);  // gyro ±2000 °/s
  ok &= writeReg(MPU_ADDR, 0x1C, 0x18);  // accel ±16 g
  delay(50);
  Serial.printf("# MPU-6050 init %s\n", ok ? "OK" : "FAILED");
  return ok;
}

bool readMpu() {
  uint8_t b[14];
  if (!readRegs(MPU_ADDR, 0x3B, b, 14)) return false;
  auto s16 = [&](int i) { return (int16_t)((b[i] << 8) | b[i + 1]); };
  ax = s16(0) / 2048.0f;  // ±16 g -> 2048 LSB/g
  ay = s16(2) / 2048.0f;
  az = s16(4) / 2048.0f;
  // Temperature formula differs: MPU-6050 vs MPU-6500-type clones (WHO_AM_I 0x70 etc.).
  tempC = mpuWho == 0x68 ? s16(6) / 340.0f + 36.53f : s16(6) / 333.87f + 21.0f;
  gx = s16(8) / 16.4f;  // ±2000 °/s -> 16.4 LSB/(°/s)
  gy = s16(10) / 16.4f;
  gz = s16(12) / 16.4f;
  return true;
}

// ---------- MAX30102 ----------
bool initMax() {
  uint8_t id;
  if (!readRegs(MAX_ADDR, 0xFF, &id, 1)) {
    Serial.println("# MAX30102: no answer at 0x57");
    return false;
  }
  Serial.printf("# MAX30102 PART_ID = 0x%02X %s\n", id, id == 0x15 ? "(ok)" : "(expected 0x15)");
  bool ok = writeReg(MAX_ADDR, 0x09, 0x40);  // reset
  delay(100);
  uint8_t status;
  readRegs(MAX_ADDR, 0x00, &status, 1);  // clear interrupt flags
  ok &= writeReg(MAX_ADDR, 0x08, 0x50);  // average 4 samples, FIFO rollover on
  ok &= writeReg(MAX_ADDR, 0x0A, 0x2B);  // 4096 nA range, 200 sps, 411 us -> 50 Hz after averaging
  ok &= writeReg(MAX_ADDR, 0x0C, 0x24);  // red LED ~7 mA
  ok &= writeReg(MAX_ADDR, 0x0D, 0x24);  // IR LED ~7 mA
  ok &= writeReg(MAX_ADDR, 0x04, 0);     // FIFO write pointer
  ok &= writeReg(MAX_ADDR, 0x05, 0);     // overflow counter
  ok &= writeReg(MAX_ADDR, 0x06, 0);     // FIFO read pointer
  ok &= writeReg(MAX_ADDR, 0x09, 0x03);  // SpO2 mode: red + IR
  Serial.printf("# MAX30102 init %s\n", ok ? "OK" : "FAILED");
  return ok;
}

// Drains the FIFO and keeps the newest sample.
bool readMax() {
  uint8_t p[3];  // 0x04 write ptr, 0x05 overflow, 0x06 read ptr
  if (!readRegs(MAX_ADDR, 0x04, p, 3)) return false;
  maxNew = (p[0] - p[2]) & 0x1F;
  for (uint8_t i = 0; i < maxNew; i++) {
    uint8_t b[6];
    if (!readRegs(MAX_ADDR, 0x07, b, 6)) return false;
    red = (((uint32_t)b[0] << 16) | (b[1] << 8) | b[2]) & 0x3FFFF;
    ir = (((uint32_t)b[3] << 16) | (b[4] << 8) | b[5]) & 0x3FFFF;
  }
  return true;
}

// Prints label + signed number with fixed decimals.
void printNum(const char *label, float v, int digits) {
  Serial.print(label);
  if (v >= 0) Serial.print('+');
  Serial.print(v, digits);
}

// ---------- main ----------
void startSensors() {
  i2cScan();
  mpuOk = initMpu();
  maxOk = initMax();
}

void printHelp() {
  Serial.println("# Sanad v0.1 sensor test. Commands: h = human view, r = raw 50 Hz CSV, s = rescan");
}

void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'h') {
      rawMode = false;
      Serial.println("# human mode");
    } else if (c == 'r') {
      rawMode = true;
      Serial.println("# t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,ir,red");
    } else if (c == 's') {
      startSensors();
    }
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  while (!Serial && millis() < 3000) delay(10);  // give the USB monitor a moment to attach

  Wire.setSDA(4);
  Wire.setSCL(5);
  Wire.begin();
  Wire.setClock(400000);

  printHelp();
  startSensors();
}

void loop() {
  handleSerial();

  // Heartbeat LED: blinking = firmware is running.
  static uint32_t lastBlink = 0;
  if (millis() - lastBlink >= 500) {
    lastBlink = millis();
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
  }

  // Missing sensor: retry every 3 s so you can fix wiring without rebooting.
  static uint32_t lastRetry = 0;
  if ((!mpuOk || !maxOk) && millis() - lastRetry >= 3000) {
    lastRetry = millis();
    if (!mpuOk) mpuOk = initMpu();
    if (!maxOk) maxOk = initMax();
  }

  // 50 Hz tick.
  static uint32_t next = micros();
  if ((int32_t)(micros() - next) < 0) return;
  next += SAMPLE_US;
  if ((int32_t)(micros() - next) > 100000) next = micros();  // fell far behind: resync

  if (mpuOk && !readMpu()) {
    mpuOk = false;
    Serial.println("# MPU-6050 LOST (loose wire?)");
  }
  if (maxOk && !readMax()) {
    maxOk = false;
    Serial.println("# MAX30102 LOST (loose wire?)");
  }

  // Floats go through Serial.print(value, digits): this core's printf prints %f as nothing.
  if (rawMode) {
    Serial.print(millis());
    const float v[6] = {ax, ay, az, gx, gy, gz};
    for (int i = 0; i < 6; i++) {
      Serial.print(',');
      Serial.print(v[i], i < 3 ? 3 : 1);
    }
    Serial.printf(",%lu,%lu\n", (unsigned long)ir, (unsigned long)red);
    return;
  }

  static uint8_t tick = 0;
  if (++tick < 25) return;  // human mode: 2 lines per second
  tick = 0;
  if (mpuOk) {
    float mag = sqrtf(ax * ax + ay * ay + az * az);
    printNum("acc[g] x=", ax, 2);
    printNum(" y=", ay, 2);
    printNum(" z=", az, 2);
    printNum(" |a|=", mag, 2);
    printNum("  gyro[dps] x=", gx, 1);
    printNum(" y=", gy, 1);
    printNum(" z=", gz, 1);
    printNum("  ", tempC, 1);
    Serial.print("C   ");
  } else {
    Serial.print("MPU-6050 MISSING   ");
  }
  if (maxOk) {
    Serial.printf("IR=%6lu red=%6lu finger=%s\n", (unsigned long)ir, (unsigned long)red,
                  ir > FINGER_IR ? "YES" : "no");
  } else {
    Serial.println("MAX30102 MISSING");
  }
}
