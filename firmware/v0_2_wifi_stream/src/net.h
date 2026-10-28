// Sanad firmware v0.2: Wi-Fi + a minimal WebSocket server (RFC 6455), no external libraries.
#pragma once
#include <Arduino.h>

// Prints "# <message>" to USB and sends it to every WebSocket client. No %f: this core's printf
// can't print floats.
void logMsg(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

// Commands received from WebSocket clients (text frames) are handed to this function.
using CommandHandler = void (*)(char);

bool netSelfTest();  // RFC 6455's example handshake; prints "# WS self-test OK" or "FAILED"
void netBegin(CommandHandler onCommand);  // saved network or own access point, then the server
void netLoop();      // call every loop(): new clients, handshakes, client frames, Wi-Fi watchdog
void netSendLine(const char *line, size_t len);  // one data line (no newline) to every client
void netPrintStatus();  // 'w' command
void netForceAp();      // 'a' command: own network "Sanad-Band" now
void netRejoin();       // 'j' command: look for the saved networks again
int netClientCount();
void netSummary(char *buf, size_t size);  // one line for the human view
