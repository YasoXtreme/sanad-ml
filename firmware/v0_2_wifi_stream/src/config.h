// Sanad firmware v0.2: settings you might want to change.
#pragma once
#include <stdint.h>

#define SANAD_VERSION "0.2.0"

// 1 = take samples from the MPU's own FIFO: its 50 Hz clock sets the timing and nothing is lost
//     while the Pico is busy for up to ~0.8 s (Wi-Fi). Gaps in `seq` show any samples that were lost.
// 0 = v0.1-style fallback: the Pico reads the newest sample every 20 ms (use it if a clone's FIFO misbehaves).
#ifndef SANAD_USE_FIFO
#define SANAD_USE_FIFO 1
#endif

// ---------- Sensors (I2C0) ----------
constexpr uint8_t PIN_SDA = 4;  // GP4 = pin 6
constexpr uint8_t PIN_SCL = 5;  // GP5 = pin 7
constexpr uint32_t I2C_HZ = 400000;
constexpr uint8_t MPU_ADDR = 0x68;  // AD0 low (default on most modules)
constexpr uint8_t MAX_ADDR = 0x57;
constexpr uint32_t SAMPLE_MS = 20;     // 50 Hz
constexpr uint32_t FINGER_IR = 50000;  // IR above this = something is on the MAX30102

// Column header of the stream (USB raw mode and WebSocket). Lines starting with '#' are comments.
constexpr char CSV_HEADER[] = "# t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,ir,red,seq";

// ---------- Wi-Fi ----------
// Saved networks (phone hotspot, home Wi-Fi) go in wifi_secrets.h: copy wifi_secrets.example.h.
struct WifiNetwork {
  const char *ssid;
  const char *password;  // "" for an open network
};
constexpr char HOSTNAME[] = "sanad-band";  // the band answers as sanad-band.local
constexpr char AP_SSID[] = "Sanad-Band";   // own network when no saved network is in range
constexpr char AP_PASSWORD[] = "sanad2026";
static_assert(sizeof(AP_PASSWORD) - 1 >= 8, "WPA2 needs a password of at least 8 characters");
constexpr uint8_t AP_CHANNEL = 6;  // 1, 6 or 11; change if the stage's 2.4 GHz band is crowded
constexpr uint8_t AP_IP[4] = {192, 168, 42, 1};
constexpr uint32_t JOIN_TIMEOUT_MS = 8000;  // per saved network
constexpr uint32_t DHCP_TIMEOUT_MS = 5000;  // after joining, to get an IP address
constexpr uint32_t LINK_LOST_MS = 15000;    // saved network gone this long -> search again / own network

// ---------- WebSocket server ----------
constexpr uint16_t HTTP_PORT = 80;
constexpr int MAX_WS_CLIENTS = 3;
constexpr int MAX_PENDING = 2;                   // connections still in the HTTP handshake
constexpr uint32_t HANDSHAKE_TIMEOUT_MS = 3000;  // unfinished handshakes are dropped after this
constexpr uint32_t STALL_CLOSE_MS = 3000;        // a client with no room for data this long is closed
constexpr uint32_t WRITE_TIMEOUT_MS = 2;         // never let a TCP write wait longer than this
