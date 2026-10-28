// Saved Wi-Fi networks for the band.
//
// 1. Copy this file to wifi_secrets.h in the same folder (src/).
// 2. Put your networks in it, up to 3. At startup the band joins the first one in this list that it
//    can see; if none is in range (or joining fails) it makes its own network "Sanad-Band".
//
// wifi_secrets.h is in .gitignore, so your passwords never go into git (the repo is public).
// Only 2.4 GHz networks work (the Pico 2 W has no 5 GHz radio). On an iPhone hotspot turn on
// "Maximise Compatibility"; on Android pick the 2.4 GHz band if the hotspot settings offer it.
#pragma once
#include "config.h"

static const WifiNetwork SAVED_NETWORKS[] = {
    {"My Phone", "hotspot-password"},
    {"Home WiFi", "home-password"},
    // {"Third network", "its-password"},
};
