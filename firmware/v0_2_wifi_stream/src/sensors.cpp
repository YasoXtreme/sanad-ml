// Sanad firmware v0.2: MPU-6050 + MAX30102 over I2C0 (SDA = GP4, SCL = GP5).
//
// New in v0.2: samples come from the MPU's FIFO (SANAD_USE_FIFO 1), so the MPU's own 50 Hz clock
// sets the timing, as in docs/device_data_format.md. Each sample gets a counter `seq`; when samples
// are lost (FIFO overflow while the Pico was busy, sensor unplugged) `seq` jumps by the number lost.

#include "sensors.h"

#include <Wire.h>

#include "config.h"
#include "net.h"  // logMsg

bool mpuOk = false;
bool maxOk = false;
uint8_t mpuWho = 0x68;
uint32_t maxIr = 0;
uint32_t maxRed = 0;

// MPU-6050 / MPU-6500 registers
constexpr uint8_t REG_SMPLRT_DIV = 0x19;
constexpr uint8_t REG_CONFIG = 0x1A;
constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
constexpr uint8_t REG_ACCEL_CONFIG2 = 0x1D;  // MPU-6500-type only (reserved on the MPU-6050)
constexpr uint8_t REG_FIFO_EN = 0x23;
constexpr uint8_t REG_INT_ENABLE = 0x38;
constexpr uint8_t REG_INT_STATUS = 0x3A;
constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;
constexpr uint8_t REG_TEMP_OUT_H = 0x41;
constexpr uint8_t REG_USER_CTRL = 0x6A;
constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
constexpr uint8_t REG_FIFO_COUNT_H = 0x72;
constexpr uint8_t REG_FIFO_R_W = 0x74;
constexpr uint8_t REG_WHO_AM_I = 0x75;

constexpr uint8_t SAMPLE_BYTES = 12;  // accel xyz + gyro xyz, 2 bytes each
constexpr uint32_t POLL_US = 5000;    // how often the FIFO is checked (t_ms is good to ~5 ms)

static uint32_t nextSeq = 0;       // seq of the next sample handed out
static uint32_t lastSampleMs = 0;  // t_ms of the last sample handed out (or of the last FIFO restart)
static uint32_t lastPollUs = 0;
static bool morePending = false;   // FIFO held more than one batch at the last poll
static uint8_t oddCounts = 0;      // FIFO count not a multiple of 12 this many polls in a row

// ---------- I2C helpers (from v0.1) ----------
static bool writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool readRegs(uint8_t addr, uint8_t reg, uint8_t *buf, size_t n) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, n) != n) return false;
  for (size_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

static void i2cScan() {
  char line[160];
  int len = snprintf(line, sizeof(line), "I2C scan:");
  int found = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0 && len < (int)sizeof(line) - 8) {
      len += snprintf(line + len, sizeof(line) - len, " 0x%02X", a);
      found++;
    }
  }
  logMsg("%s%s   (expected: 0x57 = MAX30102, 0x68 = MPU-6050)", line,
         found ? "" : " NOTHING FOUND -> check 3V3, GND, SDA=GP4, SCL=GP5");
}

// ---------- MPU-6050 ----------
static bool initMpu() {
  uint8_t who;
  if (!readRegs(MPU_ADDR, REG_WHO_AM_I, &who, 1)) {
    logMsg("MPU-6050: no answer at 0x68");
    return false;
  }
  mpuWho = who;
  logMsg("MPU-6050 WHO_AM_I = 0x%02X %s", who,
         who == 0x68 ? "(ok)" : "(not 0x68: MPU-6500-type clone, works the same)");
  bool ok = writeReg(MPU_ADDR, REG_PWR_MGMT_1, 0x80);  // reset
  delay(100);
  ok &= writeReg(MPU_ADDR, REG_PWR_MGMT_1, 0x01);  // wake up, clock = gyro X PLL
  ok &= writeReg(MPU_ADDR, REG_CONFIG, 0x04);      // DLPF_CFG = 4: ~20 Hz low-pass (gyro, and accel on a 6050)
  if (who != 0x68) {
    ok &= writeReg(MPU_ADDR, REG_ACCEL_CONFIG2, 0x04);  // clones filter the accel separately: same ~20 Hz
  }
  ok &= writeReg(MPU_ADDR, REG_SMPLRT_DIV, 19);       // 1 kHz / (1 + 19) = 50 Hz
  ok &= writeReg(MPU_ADDR, REG_GYRO_CONFIG, 0x18);    // gyro ±2000 °/s
  ok &= writeReg(MPU_ADDR, REG_ACCEL_CONFIG, 0x18);   // accel ±16 g
#if SANAD_USE_FIFO
  ok &= writeReg(MPU_ADDR, REG_INT_ENABLE, 0x10);  // flag FIFO overflow in INT_STATUS (INT pin unused)
  ok &= writeReg(MPU_ADDR, REG_FIFO_EN, 0x78);     // FIFO gets accel xyz + gyro xyz = 12 bytes/sample
#endif
  delay(50);
  logMsg("MPU-6050 init %s (%s)", ok ? "OK" : "FAILED",
         SANAD_USE_FIFO ? "FIFO, MPU clock" : "timer poll, Pico clock");
  return ok;
}

// Empties the FIFO and starts filling it again. Samples produced since the last one handed out
// are thrown away, and `seq` skips over them so the gap is visible downstream.
static void fifoRestart(bool countLost) {
  uint32_t now = millis();
  if (countLost) nextSeq += (now - lastSampleMs) / SAMPLE_MS;
  lastSampleMs = now;
  oddCounts = 0;
  morePending = false;
#if SANAD_USE_FIFO
  writeReg(MPU_ADDR, REG_USER_CTRL, 0x00);  // FIFO off
  writeReg(MPU_ADDR, REG_USER_CTRL, 0x04);  // FIFO reset (bit clears itself)
  writeReg(MPU_ADDR, REG_USER_CTRL, 0x40);  // FIFO on
  uint8_t status;
  readRegs(MPU_ADDR, REG_INT_STATUS, &status, 1);  // clears an old overflow flag
#endif
}

float mpuTempC() {
  uint8_t b[2];
  if (!mpuOk || !readRegs(MPU_ADDR, REG_TEMP_OUT_H, b, 2)) return NAN;
  int16_t t = (int16_t)((b[0] << 8) | b[1]);
  // The formula differs: MPU-6050 vs MPU-6500-type clones.
  return mpuWho == 0x68 ? t / 340.0f + 36.53f : t / 333.87f + 21.0f;
}

static void mpuLost() {
  mpuOk = false;
  logMsg("MPU-6050 LOST (loose wire?)");
}

// ---------- MAX30102 (from v0.1) ----------
static bool initMax() {
  uint8_t id;
  if (!readRegs(MAX_ADDR, 0xFF, &id, 1)) {
    logMsg("MAX30102: no answer at 0x57");
    return false;
  }
  logMsg("MAX30102 PART_ID = 0x%02X %s", id, id == 0x15 ? "(ok)" : "(expected 0x15)");
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
  logMsg("MAX30102 init %s", ok ? "OK" : "FAILED");
  return ok;
}

// Drains the MAX30102 FIFO and keeps the newest sample.
static bool readMax() {
  uint8_t p[3];  // 0x04 write ptr, 0x05 overflow, 0x06 read ptr
  if (!readRegs(MAX_ADDR, 0x04, p, 3)) return false;
  uint8_t fresh = (p[0] - p[2]) & 0x1F;
  for (uint8_t i = 0; i < fresh; i++) {
    uint8_t b[6];
    if (!readRegs(MAX_ADDR, 0x07, b, 6)) return false;
    maxRed = (((uint32_t)b[0] << 16) | (b[1] << 8) | b[2]) & 0x3FFFF;
    maxIr = (((uint32_t)b[3] << 16) | (b[4] << 8) | b[5]) & 0x3FFFF;
  }
  return true;
}

static void pollMax() {
  if (maxOk && !readMax()) {
    maxOk = false;
    logMsg("MAX30102 LOST (loose wire?)");
  }
}

// ---------- public ----------
void sensorsBegin() {
  Wire.setSDA(PIN_SDA);
  Wire.setSCL(PIN_SCL);
  Wire.begin();
  Wire.setClock(I2C_HZ);
  sensorsStart();
}

void sensorsStart() {
  i2cScan();
  mpuOk = initMpu();
  maxOk = initMax();
  if (mpuOk) fifoRestart(true);
}

void sensorsRetry() {
  static uint32_t lastRetry = 0;
  if ((mpuOk && maxOk) || millis() - lastRetry < 3000) return;
  lastRetry = millis();
  if (!mpuOk) {
    mpuOk = initMpu();
    if (mpuOk) fifoRestart(true);  // the time it was missing shows as a seq gap
  }
  if (!maxOk) maxOk = initMax();
}

void sensorsRestartStream() {
  nextSeq = 0;
  fifoRestart(false);
  lastPollUs = micros();
}

#if SANAD_USE_FIFO

int sensorsPoll(Sample *out, int maxOut) {
  if (!mpuOk) return 0;
  uint32_t nowUs = micros();
  if (!morePending && nowUs - lastPollUs < POLL_US) return 0;
  lastPollUs = nowUs;
  morePending = false;

  uint8_t status, c[2];
  if (!readRegs(MPU_ADDR, REG_INT_STATUS, &status, 1) || !readRegs(MPU_ADDR, REG_FIFO_COUNT_H, c, 2)) {
    mpuLost();
    return 0;
  }
  uint32_t now = millis();
  uint16_t count = ((c[0] << 8) | c[1]) & 0x1FFF;
  // A full FIFO overwrites its oldest bytes; the clone's 512 bytes are not a multiple of 12, so after
  // an overflow the samples no longer line up. Throw everything away and start again.
  uint16_t capacity = mpuWho == 0x68 ? 1024 : 512;
  bool misaligned = (count % SAMPLE_BYTES) != 0 && ++oddCounts >= 2;
  if ((count % SAMPLE_BYTES) == 0) oddCounts = 0;
  if ((status & 0x10) || count > capacity - SAMPLE_BYTES || misaligned) {
    uint32_t lost = (now - lastSampleMs) / SAMPLE_MS;
    fifoRestart(true);
    logMsg("MPU FIFO %s: ~%lu samples lost (Pico was busy), seq skips them",
           misaligned ? "out of step" : "overflow", (unsigned long)lost);
    return 0;
  }

  int queued = count / SAMPLE_BYTES;
  int n = queued;
  if (n > maxOut) n = maxOut;
  if (n > MAX_BATCH) n = MAX_BATCH;
  if (n == 0) return 0;

  uint8_t buf[MAX_BATCH * SAMPLE_BYTES];
  if (!readRegs(MPU_ADDR, REG_FIFO_R_W, buf, n * SAMPLE_BYTES)) {
    mpuLost();
    return 0;
  }
  for (int i = 0; i < n; i++) {
    const uint8_t *b = buf + i * SAMPLE_BYTES;
    out[i].seq = nextSeq++;
    out[i].t_ms = now - (uint32_t)(queued - 1 - i) * SAMPLE_MS;  // older samples waited in the queue
    for (int k = 0; k < 6; k++) out[i].raw[k] = (int16_t)((b[2 * k] << 8) | b[2 * k + 1]);
  }
  lastSampleMs = out[n - 1].t_ms;
  morePending = queued > n;
  pollMax();  // once per batch (normally one sample): keeps the newest red/IR values
  return n;
}

#else  // v0.1-style: the Pico's timer decides when to read the newest sample

int sensorsPoll(Sample *out, int maxOut) {
  if (!mpuOk || maxOut < 1) return 0;
  static uint32_t nextUs = micros();
  if ((int32_t)(micros() - nextUs) < 0) return 0;
  nextUs += SAMPLE_MS * 1000;
  if ((int32_t)(micros() - nextUs) > 100000) {  // fell far behind: resync, count the missed ticks
    nextSeq += (micros() - nextUs) / (SAMPLE_MS * 1000);
    nextUs = micros();
  }
  uint8_t b[14];
  if (!readRegs(MPU_ADDR, REG_ACCEL_XOUT_H, b, 14)) {
    mpuLost();
    return 0;
  }
  out[0].seq = nextSeq++;
  out[0].t_ms = millis();
  for (int k = 0; k < 3; k++) out[0].raw[k] = (int16_t)((b[2 * k] << 8) | b[2 * k + 1]);
  for (int k = 3; k < 6; k++) out[0].raw[k] = (int16_t)((b[2 * k + 2] << 8) | b[2 * k + 3]);  // skip temp
  lastSampleMs = out[0].t_ms;
  pollMax();
  return 1;
}

#endif
