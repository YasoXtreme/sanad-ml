"""Serve the Sanad visualizer at http://localhost:8000 (works with no internet).

Run from the repo root:  uv run python -I tools/visualizer/serve.py
Web Serial (USB) needs a secure context, and http://localhost counts as one.

Phone alerts (proof of concept): every impact flash also POSTs to /alert, and this server forwards it to
ntfy.sh as a push notification. Install the free "ntfy" app on the phone and subscribe to the topic printed
at startup. The topic is random and kept in tools/visualizer/.alert_topic (not in git): anyone who knows it can
read the alerts. --no-alerts turns this off. Needs internet on the laptop.
"""

import argparse
import functools
import http.server
import json
import secrets
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import ClassVar

HERE = Path(__file__).resolve().parent
TOPIC_FILE = HERE / ".alert_topic"
ALERT_COOLDOWN_S = 10  # one fall shakes the band several times: send one notification, not five


class Alerts:
    def __init__(self, topic: str | None):
        self.topic = topic
        self.last = -1e9
        self.lock = threading.Lock()

    def send(self, peak_g: float) -> str:
        if not self.topic:
            return "off"
        with self.lock:
            now = time.monotonic()
            if now - self.last < ALERT_COOLDOWN_S:
                return "cooldown"
            self.last = now
        clock = time.strftime("%H:%M:%S")
        req = urllib.request.Request(
            f"https://ntfy.sh/{self.topic}",
            data=f"Impact of {peak_g:.1f} g at {clock}. (Test: trigger only, no fall model yet.)".encode(),
            headers={"Title": "Sanad: possible fall", "Priority": "high", "Tags": "rotating_light"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                ok = 200 <= r.status < 300
        except OSError as e:
            print(f"alert failed: {e}", flush=True)
            return "failed"
        print(f"alert sent ({peak_g:.1f} g)" if ok else "alert failed", flush=True)
        return "sent" if ok else "failed"


def load_topic() -> str:
    if TOPIC_FILE.exists():
        topic = TOPIC_FILE.read_text(encoding="utf-8").strip()
        if topic:
            return topic
    topic = f"sanad-{secrets.token_hex(5)}"
    TOPIC_FILE.write_text(topic + "\n", encoding="utf-8")
    return topic


class Server(http.server.ThreadingHTTPServer):
    # On Windows, SO_REUSEADDR lets a second server bind a port that is already in use, so an old copy
    # could keep answering. Without it, a busy port fails and we move on to the next one.
    allow_reuse_address = sys.platform != "win32"


class Handler(http.server.SimpleHTTPRequestHandler):
    alerts: ClassVar[Alerts] = Alerts(None)

    # Windows' registry sometimes maps .js to text/plain, and browsers refuse ES modules served that way.
    extensions_map: ClassVar[dict[str, str]] = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html",
        ".js": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".csv": "text/csv",
        ".txt": "text/plain",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # edits show up on reload
        super().end_headers()

    def do_POST(self):
        if self.path != "/alert":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            peak_g = float(json.loads(self.rfile.read(n) or b"{}").get("peakG", 0))
        except (ValueError, TypeError, AttributeError):
            peak_g = 0.0
        body = json.dumps({"status": self.alerts.send(peak_g)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_request(self, code="-", size="-"):
        if isinstance(code, int) and code >= 400:
            super().log_request(code, size)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    ap.add_argument(
        "--no-alerts", action="store_true", help="don't send phone notifications on impacts"
    )
    args = ap.parse_args()

    if not args.no_alerts:
        topic = load_topic()
        Handler.alerts = Alerts(topic)
        print(
            f"Phone alerts: in the ntfy app subscribe to '{topic}'  (or open https://ntfy.sh/{topic})"
        )

    handler = functools.partial(Handler, directory=str(HERE))
    for port in range(args.port, args.port + 10):
        try:
            server = Server(("127.0.0.1", port), handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"ports {args.port}-{args.port + 9} are all busy")

    url = f"http://localhost:{port}/"
    print(f"Sanad visualizer: {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
