// Sanad firmware v0.2: MPU-6050 + MAX30102 over I2C0, plain register access (no libraries).
#pragma once
#include <Arduino.h>

// One MPU sample. Axes are the chip's own axes, not yet rotated into the Sanad frame.
struct Sample {
  uint32_t seq;   // sample counter since start; a gap = samples that were lost
  uint32_t t_ms;  // millis() when the MPU took the sample (estimated from the FIFO queue)
  int16_t raw[6]; // ax, ay, az (2048 counts/g), gx, gy, gz (16.4 counts per °/s)
};

constexpr int MAX_BATCH = 20;  // samples per poll: 20 x 12 bytes = 240, fits Wire's 256-byte buffer

extern bool mpuOk;
extern bool maxOk;
extern uint8_t mpuWho;   // WHO_AM_I: 0x68 = genuine MPU-6050, 0x70 etc. = MPU-6500-type clone
extern uint32_t maxIr;   // newest MAX30102 values (18-bit counts)
extern uint32_t maxRed;

void sensorsBegin();          // I2C bus, scan, both sensors
void sensorsStart();          // rescan the bus and re-initialise both sensors ('s' command)
void sensorsRetry();          // call often: re-initialises a missing sensor every 3 s
void sensorsRestartStream();  // empty the MPU FIFO and restart seq at 0 (end of setup)
int sensorsPoll(Sample *out, int maxOut);  // new samples since the last call, oldest first
float mpuTempC();             // reads the temperature register now

inline float accelG(int16_t counts) { return counts / 2048.0f; }  // ±16 g range
inline float gyroDps(int16_t counts) { return counts / 16.4f; }   // ±2000 °/s range
