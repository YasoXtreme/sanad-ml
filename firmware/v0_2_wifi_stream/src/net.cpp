// Sanad firmware v0.2: Wi-Fi and a minimal WebSocket server (RFC 6455), no external libraries.
//
// Wi-Fi: join the first saved network (wifi_secrets.h) that is in range; otherwise make our own
// network "Sanad-Band" at 192.168.42.1. If the saved network is gone for 15 s, do it all again.
//
// Server on port 80:
//   GET /ws with "Upgrade: websocket" -> WebSocket: every sample is one text message (one CSV line)
//   GET /                             -> small HTML status page
// Sending never waits: a frame goes out only if the TCP buffer has room for all of it, otherwise
// that line is dropped for that client (the seq gap shows it). No room for 3 s -> client closed.

#include "net.h"

#include <Hash.h>
#include <SimpleMDNS.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"  // inside the core's WiFiClient.h, not ours
#include <WiFi.h>
#pragma GCC diagnostic pop
#include <libb64/cencode.h>
#include <pico/cyw43_arch.h>
#include <stdarg.h>

#include "config.h"
#include "sensors.h"

#if __has_include("wifi_secrets.h")
#include "wifi_secrets.h"
#define HAVE_SAVED_NETWORKS 1
static_assert(sizeof(SAVED_NETWORKS) / sizeof(SAVED_NETWORKS[0]) <= 3, "wifi_secrets.h: at most 3 networks");
#else
#define HAVE_SAVED_NETWORKS 0
#endif

namespace {

enum class Mode : uint8_t { Off, Station, AccessPoint };

constexpr char WS_GUID[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
constexpr uint8_t OP_CONT = 0x0;
constexpr uint8_t OP_TEXT = 0x1;
constexpr uint8_t OP_CLOSE = 0x8;
constexpr uint8_t OP_PING = 0x9;
constexpr uint8_t OP_PONG = 0xA;
constexpr size_t MAX_PAYLOAD = 240;  // data lines are < 90 bytes, log lines < 220

struct WsClient {
  WiFiClient sock;
  bool active = false;
  bool broken = false;    // half a frame went out: the stream is corrupt, close it
  bool dropping = false;  // the last frame found no room
  uint32_t dropSinceMs = 0;
  uint32_t sent = 0;
  uint32_t dropped = 0;
  uint8_t rx[160] = {};  // frames from the client (commands, ping, close) are tiny
  uint16_t rxLen = 0;
};

struct PendingClient {  // connection still sending its HTTP request
  WiFiClient sock;
  bool active = false;
  uint32_t startMs = 0;
  char line[128] = {};  // current header line; longer lines are cut (we only need short ones)
  uint8_t lineLen = 0;
  uint8_t lines = 0;
  bool get = false;
  bool pathWs = false;
  bool pathRoot = false;
  bool upgrade = false;
  char key[64] = {};
};

Mode mode = Mode::Off;
char joinedSsid[33] = "";
bool linkLost = false;
uint32_t linkLostSinceMs = 0;
WiFiServer server(HTTP_PORT);
bool serverStarted = false;
bool mdnsStarted = false;
CommandHandler commandHandler = nullptr;
char commandQueue[8];
uint8_t commandCount = 0;
uint32_t sentTotal = 0;
uint32_t droppedTotal = 0;
WsClient ws[MAX_WS_CLIENTS];
PendingClient pending[MAX_PENDING];

// ---------- WebSocket frames ----------
// Server frames are never masked. Our payloads are short, so the header is 2 or 4 bytes.
size_t buildFrame(uint8_t *out, uint8_t opcode, const void *payload, size_t len) {
  size_t h = 0;
  out[h++] = 0x80 | opcode;  // FIN + opcode
  if (len < 126) {
    out[h++] = (uint8_t)len;
  } else {
    out[h++] = 126;
    out[h++] = (uint8_t)(len >> 8);
    out[h++] = (uint8_t)(len & 0xFF);
  }
  memcpy(out + h, payload, len);
  return h + len;
}

// Header + payload in ONE write, and only if the TCP buffer can take all of it: never waits.
bool writeFrame(WsClient &c, const uint8_t *frame, size_t n) {
  if (!c.active || c.broken) return false;
  if ((size_t)c.sock.availableForWrite() >= n) {
    size_t k = c.sock.write(frame, n);
    if (k == n) {
      c.dropping = false;
      return true;
    }
    if (k > 0) {
      c.broken = true;
      return false;
    }
  }
  if (!c.dropping) {
    c.dropping = true;
    c.dropSinceMs = millis();
  }
  return false;
}

void sendText(WsClient &c, const char *text) {
  size_t len = strlen(text);
  if (len > MAX_PAYLOAD) len = MAX_PAYLOAD;
  uint8_t frame[4 + MAX_PAYLOAD];
  writeFrame(c, frame, buildFrame(frame, OP_TEXT, text, len));
}

void broadcastText(const char *text, size_t len) {
  if (len > MAX_PAYLOAD) len = MAX_PAYLOAD;
  uint8_t frame[4 + MAX_PAYLOAD];
  size_t n = 0;
  for (WsClient &c : ws) {
    if (!c.active) continue;
    if (!n) n = buildFrame(frame, OP_TEXT, text, len);
    writeFrame(c, frame, n);
  }
}

// Sec-WebSocket-Accept = base64(sha1(key + GUID))
void wsAcceptKey(const char *key, char out[32]) {
  char buf[sizeof(PendingClient::key) + sizeof(WS_GUID)];
  int n = snprintf(buf, sizeof(buf), "%s%s", key, WS_GUID);
  uint8_t digest[20];
  sha1(buf, (uint32_t)n, digest);
  base64_encode_chars((const char *)digest, 20, out);
}

// ---------- clients ----------
void closeWs(WsClient &c, const char *why) {
  if (!c.active) return;
  c.active = false;
  String ip = c.sock.remoteIP().toString();
  c.sock.stop(20);
  c.sock = WiFiClient();
  logMsg("WebSocket client %s left (%s): sent %lu lines, dropped %lu", ip.c_str(), why,
         (unsigned long)c.sent, (unsigned long)c.dropped);
}

void dropPending(PendingClient &p) {
  while (p.sock.available() > 0) p.sock.read();  // unread data makes lwIP reset instead of close
  p.sock.stop(50);
  p.sock = WiFiClient();
  p.active = false;
}

void closeAllClients() {
  for (WsClient &c : ws) closeWs(c, "Wi-Fi is changing");
  for (PendingClient &p : pending) {
    if (p.active) dropPending(p);
  }
}

String ipString() {
  if (mode == Mode::Station) return WiFi.localIP().toString();
  if (mode == Mode::AccessPoint) return WiFi.softAPIP().toString();
  return String("(none)");
}

void sendHttp(WiFiClient &sock, const char *status, const char *type, const char *body) {
  char head[200];
  int n = snprintf(head, sizeof(head),
                   "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %u\r\n"
                   "Cache-Control: no-store\r\nConnection: close\r\n\r\n",
                   status, type, (unsigned)strlen(body));
  sock.write((const uint8_t *)head, n);
  sock.write((const uint8_t *)body, strlen(body));
}

void sendStatusPage(WiFiClient &sock) {
  static char body[1600];
  String ip = ipString();
  char network[96];
  if (mode == Mode::Station) {
    snprintf(network, sizeof(network), "joined '%s'", joinedSsid);
  } else if (mode == Mode::AccessPoint) {
    snprintf(network, sizeof(network), "own network '%s' (password %s)", AP_SSID, AP_PASSWORD);
  } else {
    snprintf(network, sizeof(network), "off");
  }
  char mpu[48] = "MISSING";
  if (mpuOk) snprintf(mpu, sizeof(mpu), "ok (WHO_AM_I 0x%02X)", mpuWho);
  snprintf(body, sizeof(body),
           "<!doctype html><html><head><meta charset=utf-8>"
           "<meta name=viewport content=\"width=device-width\"><meta http-equiv=refresh content=5>"
           "<title>Sanad band</title><style>body{font-family:sans-serif;margin:2em;max-width:42em}"
           "td{padding:.25em 1.5em .25em 0;vertical-align:top}code{background:#eee;padding:.1em .3em}"
           "</style></head><body><h1>Sanad band</h1><table>"
           "<tr><td>Firmware</td><td>v%s</td></tr>"
           "<tr><td>Wi-Fi</td><td>%s</td></tr>"
           "<tr><td>Address</td><td>%s (also %s.local)</td></tr>"
           "<tr><td>Visualizer</td><td><code>ws://%s/ws</code></td></tr>"
           "<tr><td>Visualizers connected</td><td>%d of %d</td></tr>"
           "<tr><td>Lines sent / dropped</td><td>%lu / %lu</td></tr>"
           "<tr><td>MPU-6050</td><td>%s</td></tr>"
           "<tr><td>MAX30102</td><td>%s</td></tr>"
           "<tr><td>Uptime</td><td>%lu s</td></tr>"
           "</table><p>This page refreshes every 5 s.</p></body></html>",
           SANAD_VERSION, network, ip.c_str(), HOSTNAME, ip.c_str(), netClientCount(), MAX_WS_CLIENTS,
           (unsigned long)sentTotal, (unsigned long)droppedTotal, mpu, maxOk ? "ok" : "MISSING",
           (unsigned long)(millis() / 1000));
  sendHttp(sock, "200 OK", "text/html; charset=utf-8", body);
}

void upgradeToWs(PendingClient &p) {
  WsClient *slot = nullptr;
  for (WsClient &c : ws) {
    if (!c.active) {
      slot = &c;
      break;
    }
  }
  String ip = p.sock.remoteIP().toString();
  if (!slot) {
    sendHttp(p.sock, "503 Service Unavailable", "text/plain", "Busy: the band serves 3 visualizers at most\n");
    dropPending(p);
    logMsg("WebSocket client %s refused: already %d connected", ip.c_str(), MAX_WS_CLIENTS);
    return;
  }
  char accept[32];
  wsAcceptKey(p.key, accept);
  char resp[160];
  int n = snprintf(resp, sizeof(resp),
                   "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                   "Sec-WebSocket-Accept: %s\r\n\r\n",
                   accept);
  if (p.sock.write((const uint8_t *)resp, n) != (size_t)n) {
    dropPending(p);
    return;
  }
  *slot = WsClient();
  slot->sock = p.sock;
  slot->active = true;
  p.sock = WiFiClient();
  p.active = false;
  sendText(*slot, "# Sanad v" SANAD_VERSION " band: 50 Hz, chip axes, accel in g, gyro in deg/s; a gap in seq = lost samples");
  sendText(*slot, CSV_HEADER);
  logMsg("WebSocket client %s connected (%d of %d)", ip.c_str(), netClientCount(), MAX_WS_CLIENTS);
}

bool startsWithNoCase(const char *s, const char *prefix) {
  return strncasecmp(s, prefix, strlen(prefix)) == 0;
}

bool containsNoCase(const char *s, const char *word) {
  for (size_t n = strlen(word); *s; s++) {
    if (strncasecmp(s, word, n) == 0) return true;
  }
  return false;
}

void headerLine(PendingClient &p) {
  p.line[p.lineLen] = 0;
  if (p.lines++ == 0) {  // request line, e.g. "GET /ws HTTP/1.1"
    p.get = strncmp(p.line, "GET ", 4) == 0;
    if (!p.get) return;
    const char *path = p.line + 4;
    size_t len = strcspn(path, " ?");
    p.pathWs = len == 3 && strncmp(path, "/ws", 3) == 0;
    p.pathRoot = (len == 1 && path[0] == '/') || (len == 11 && strncmp(path, "/index.html", 11) == 0);
  } else if (startsWithNoCase(p.line, "Upgrade:")) {
    p.upgrade = containsNoCase(p.line + 8, "websocket");
  } else if (startsWithNoCase(p.line, "Sec-WebSocket-Key:")) {
    const char *v = p.line + 18;
    while (*v == ' ' || *v == '\t') v++;
    size_t len = strcspn(v, " \t");
    if (len < sizeof(p.key)) {
      memcpy(p.key, v, len);
      p.key[len] = 0;
    }
  }
}

void respond(PendingClient &p) {
  if (p.get && p.pathWs && p.upgrade && p.key[0]) {
    upgradeToWs(p);
    return;
  }
  if (p.get && p.pathRoot) {
    sendStatusPage(p.sock);
  } else if (p.get && p.pathWs) {
    sendHttp(p.sock, "400 Bad Request", "text/plain", "WebSocket only: connect to ws://<band>/ws\n");
  } else {
    sendHttp(p.sock, "404 Not Found", "text/plain", "Not found. Try / or ws://<band>/ws\n");
  }
  dropPending(p);
}

void acceptNew() {
  WiFiClient sock = server.accept();
  if (!sock) return;
  for (PendingClient &p : pending) {
    if (p.active) continue;
    p = PendingClient();
    p.sock = sock;
    p.active = true;
    p.startMs = millis();
    p.sock.setNoDelay(true);
    p.sock.setTimeout(WRITE_TIMEOUT_MS);
    return;
  }
  sock.stop(1);  // too many connections still handshaking
}

void servicePending() {
  for (PendingClient &p : pending) {
    if (!p.active) continue;
    bool done = false;
    while (!done && p.sock.available() > 0) {
      int ch = p.sock.read();
      if (ch < 0) break;
      if (ch == '\n') {
        if (p.lineLen == 0 && p.lines > 0) {
          done = true;  // blank line: end of the request headers
        } else if (p.lineLen > 0) {
          headerLine(p);
        }
        p.lineLen = 0;
      } else if (ch != '\r' && p.lineLen < sizeof(p.line) - 1) {
        p.line[p.lineLen++] = (char)ch;
      }
    }
    if (done) {
      respond(p);
    } else if (!p.sock.connected() || millis() - p.startMs > HANDSHAKE_TIMEOUT_MS) {
      dropPending(p);
    }
  }
}

void queueCommand(char ch) {
  if (ch == ' ' || ch == '\r' || ch == '\n' || ch == '\t') return;
  if (commandCount < sizeof(commandQueue)) commandQueue[commandCount++] = ch;
}

// Handles complete frames in c.rx. Returns false if the client was closed.
bool parseFrames(WsClient &c) {
  while (c.rxLen >= 2) {
    uint8_t opcode = c.rx[0] & 0x0F;
    bool masked = c.rx[1] & 0x80;  // clients must mask; we accept both
    size_t len = c.rx[1] & 0x7F;
    size_t h = 2;
    if (len == 126) {
      if (c.rxLen < 4) return true;
      len = ((size_t)c.rx[2] << 8) | c.rx[3];
      h = 4;
    } else if (len == 127) {
      closeWs(c, "frame too big");
      return false;
    }
    size_t maskAt = h;
    if (masked) h += 4;
    if (h + len > sizeof(c.rx)) {
      closeWs(c, "frame too big");
      return false;
    }
    if (c.rxLen < h + len) return true;  // wait for the rest
    uint8_t *payload = c.rx + h;
    if (masked) {
      for (size_t i = 0; i < len; i++) payload[i] ^= c.rx[maskAt + (i & 3)];
    }
    if (opcode == OP_TEXT || opcode == OP_CONT) {  // commands: the same letters as on USB
      for (size_t i = 0; i < len; i++) queueCommand((char)payload[i]);
    } else if (opcode == OP_PING) {
      uint8_t frame[4 + 125];
      writeFrame(c, frame, buildFrame(frame, OP_PONG, payload, len > 125 ? 125 : len));
    } else if (opcode == OP_CLOSE) {
      uint8_t frame[4 + 2];
      writeFrame(c, frame, buildFrame(frame, OP_CLOSE, payload, len >= 2 ? 2 : 0));  // echo the code
      closeWs(c, "closed by the visualizer");
      return false;
    }  // binary and pong frames are ignored
    size_t used = h + len;
    memmove(c.rx, c.rx + used, c.rxLen - used);
    c.rxLen -= used;
  }
  return true;
}

void serviceClients() {
  for (WsClient &c : ws) {
    if (!c.active) continue;
    if (c.broken) {
      closeWs(c, "send error");
      continue;
    }
    if (c.dropping && millis() - c.dropSinceMs > STALL_CLOSE_MS) {
      closeWs(c, "no room for 3 s, link too slow");
      continue;
    }
    while (c.rxLen < sizeof(c.rx)) {
      int avail = c.sock.available();
      if (avail <= 0) break;
      size_t room = sizeof(c.rx) - c.rxLen;
      int k = c.sock.read(c.rx + c.rxLen, (size_t)avail < room ? (size_t)avail : room);
      if (k <= 0) break;
      c.rxLen += k;
    }
    if (!parseFrames(c)) continue;
    if (!c.sock.connected()) closeWs(c, "connection closed");
  }
}

// ---------- Wi-Fi ----------
bool staLinkUp() {
  return cyw43_wifi_link_status(&cyw43_state, CYW43_ITF_STA) == CYW43_LINK_JOIN && WiFi.localIP().isSet();
}

void stopWifi() {
  if (mode == Mode::Station) WiFi.disconnect();
  if (mode == Mode::AccessPoint) WiFi.disconnectAP();
  mode = Mode::Off;
}

#if HAVE_SAVED_NETWORKS
// Scan results are not always zero-terminated at 32 characters.
bool ssidMatches(const char *seen, const char *wanted) {
  size_t n = strlen(wanted);
  if (!seen || n == 0 || n > 32) return false;
  return strncmp(seen, wanted, n) == 0 && (n == 32 || seen[n] == '\0');
}
#endif

bool joinSaved() {
#if HAVE_SAVED_NETWORKS
  stopWifi();
  WiFi.mode(WIFI_STA);  // also makes WiFi.begin() return without waiting for DHCP (we wait below)
  WiFi.setHostname(HOSTNAME);
  WiFi.setTimeout(JOIN_TIMEOUT_MS);
  logMsg("Wi-Fi: looking for saved networks ...");
  constexpr size_t COUNT = sizeof(SAVED_NETWORKS) / sizeof(SAVED_NETWORKS[0]);
  int seen = WiFi.scanNetworks();
  bool visible[COUNT] = {};
  bool anyVisible = false;
  for (size_t k = 0; k < COUNT; k++) {  // check the scan first: a join attempt may clear it
    for (int i = 0; i < seen && !visible[k]; i++) {
      visible[k] = ssidMatches(WiFi.SSID((uint8_t)i), SAVED_NETWORKS[k].ssid);
    }
    anyVisible |= visible[k];
  }
  WiFi.scanDelete();
  for (size_t k = 0; k < COUNT; k++) {
    if (!visible[k]) continue;
    const WifiNetwork &net = SAVED_NETWORKS[k];
    logMsg("Wi-Fi: joining '%s' ...", net.ssid);
    bool open = !net.password || !net.password[0];
    int status = open ? WiFi.begin(net.ssid) : WiFi.begin(net.ssid, net.password);
    uint32_t t0 = millis();
    while (status != WL_IDLE_STATUS && !staLinkUp() && millis() - t0 < DHCP_TIMEOUT_MS) delay(20);
    if (staLinkUp()) {
      mode = Mode::Station;
      snprintf(joinedSsid, sizeof(joinedSsid), "%s", net.ssid);
      return true;
    }
    logMsg("Wi-Fi: could not join '%s' (wrong password, WPA3 only, or weak signal?)", net.ssid);
    WiFi.disconnect();
    cyw43_wifi_leave(&cyw43_state, CYW43_ITF_STA);  // a failed join keeps retrying in the chip otherwise
  }
  if (!anyVisible) {
    logMsg("Wi-Fi: none of the %u saved networks is in range (%d networks seen)", (unsigned)COUNT, seen);
  }
#else
  logMsg("Wi-Fi: no saved networks (copy src/wifi_secrets.example.h to src/wifi_secrets.h)");
#endif
  return false;
}

bool startAp() {
  stopWifi();
  WiFi.mode(WIFI_AP);  // softAPConfig() only stores the address in AP mode
  IPAddress ip(AP_IP[0], AP_IP[1], AP_IP[2], AP_IP[3]);
  WiFi.softAPConfig(ip, ip, IPAddress(255, 255, 255, 0));
  if (WiFi.beginAP(AP_SSID, AP_PASSWORD, AP_CHANNEL) != WL_CONNECTED) {
    logMsg("Wi-Fi: could not start own network '%s' (USB still works)", AP_SSID);
    return false;
  }
  mode = Mode::AccessPoint;
  return true;
}

void startServices() {
  if (!serverStarted) {
    server.setNoDelay(true);
    server.begin();
    serverStarted = true;
  }
  if (!mdnsStarted && mode != Mode::Off) mdnsStarted = MDNS.begin(HOSTNAME);  // follows later Wi-Fi changes
}

void connectWifi() {
  closeAllClients();
  if (!joinSaved()) startAp();
  startServices();
  linkLost = false;
}

void watchLink() {
  static uint32_t lastCheck = 0;
  if (mode != Mode::Station || millis() - lastCheck < 500) return;
  lastCheck = millis();
  if (staLinkUp()) {
    if (linkLost) logMsg("Wi-Fi: link to '%s' is back", joinedSsid);
    linkLost = false;
    return;
  }
  if (!linkLost) {
    linkLost = true;
    linkLostSinceMs = millis();
    logMsg("Wi-Fi: link to '%s' lost, waiting up to %lu s", joinedSsid, (unsigned long)(LINK_LOST_MS / 1000));
    return;
  }
  if (millis() - linkLostSinceMs >= LINK_LOST_MS) {
    logMsg("Wi-Fi: still no link, searching again (samples pause while joining; seq shows the gap)");
    connectWifi();
    netPrintStatus();
  }
}

void runCommands() {
  // Runs after all clients were serviced, because a command ('a', 'j') may close them all.
  char cmds[sizeof(commandQueue)];
  uint8_t n = commandCount;
  memcpy(cmds, commandQueue, n);
  commandCount = 0;
  for (uint8_t i = 0; i < n; i++) {
    if (commandHandler) commandHandler(cmds[i]);
  }
}

}  // namespace

// ---------- public ----------
void logMsg(const char *fmt, ...) {
  char buf[224] = "# ";
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf + 2, sizeof(buf) - 2, fmt, ap);
  va_end(ap);
  if (n < 0) return;
  Serial.println(buf);
  broadcastText(buf, strlen(buf));
}

bool netSelfTest() {
  char out[32];
  wsAcceptKey("dGhlIHNhbXBsZSBub25jZQ==", out);  // the example in RFC 6455, section 1.3
  bool ok = strcmp(out, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") == 0;
  if (ok) {
    logMsg("WS self-test OK");
  } else {
    logMsg("WS self-test FAILED (got %s)", out);
  }
  return ok;
}

void netBegin(CommandHandler onCommand) {
  commandHandler = onCommand;
  netSelfTest();
  connectWifi();
  netPrintStatus();
}

void netLoop() {
  if (!serverStarted) return;
  watchLink();
  acceptNew();
  servicePending();
  serviceClients();
  runCommands();
}

void netSendLine(const char *line, size_t len) {
  if (len > MAX_PAYLOAD) return;
  uint8_t frame[4 + MAX_PAYLOAD];
  size_t n = 0;
  for (WsClient &c : ws) {
    if (!c.active || c.broken) continue;
    if (!n) n = buildFrame(frame, OP_TEXT, line, len);
    if (writeFrame(c, frame, n)) {
      c.sent++;
      sentTotal++;
    } else {
      c.dropped++;
      droppedTotal++;
    }
  }
}

int netClientCount() {
  int n = 0;
  for (const WsClient &c : ws) n += c.active;
  return n;
}

void netPrintStatus() {
  String ip = ipString();
  if (mode == Mode::Station) {
    logMsg("Wi-Fi: joined '%s', band at %s (or %s.local), signal %ld dBm", joinedSsid, ip.c_str(), HOSTNAME,
           (long)WiFi.RSSI());
  } else if (mode == Mode::AccessPoint) {
    logMsg("Wi-Fi: own network '%s', password '%s', channel %u, band at %s, %u device(s) joined", AP_SSID,
           AP_PASSWORD, (unsigned)AP_CHANNEL, ip.c_str(), (unsigned)WiFi.softAPgetStationNum());
  } else {
    logMsg("Wi-Fi: off (could not start; USB still works)");
  }
  if (mode != Mode::Off) logMsg("Visualizer: ws://%s/ws   status page: http://%s/", ip.c_str(), ip.c_str());
  logMsg("Visualizers connected: %d of %d, lines sent %lu, dropped %lu", netClientCount(), MAX_WS_CLIENTS,
         (unsigned long)sentTotal, (unsigned long)droppedTotal);
}

void netForceAp() {
  if (mode == Mode::AccessPoint) {
    logMsg("Wi-Fi: already on own network '%s'", AP_SSID);
    return;
  }
  logMsg("Wi-Fi: switching to own network '%s' now", AP_SSID);
  closeAllClients();
  startAp();
  startServices();
  linkLost = false;
  netPrintStatus();
}

void netRejoin() {
  if (!HAVE_SAVED_NETWORKS) {
    logMsg("Wi-Fi: no saved networks (copy src/wifi_secrets.example.h to src/wifi_secrets.h)");
    return;
  }
  logMsg("Wi-Fi: looking for saved networks again (samples pause while joining)");
  connectWifi();
  netPrintStatus();
}

void netSummary(char *buf, size_t size) {
  String ip = ipString();
  const char *m = mode == Mode::Station ? "joined" : mode == Mode::AccessPoint ? "own network" : "off";
  const char *ssid = mode == Mode::Station ? joinedSsid : mode == Mode::AccessPoint ? AP_SSID : "";
  snprintf(buf, size, "Wi-Fi: %s '%s' at %s   visualizers %d/%d   lines sent %lu, dropped %lu", m, ssid, ip.c_str(),
           netClientCount(), MAX_WS_CLIENTS, (unsigned long)sentTotal, (unsigned long)droppedTotal);
}
