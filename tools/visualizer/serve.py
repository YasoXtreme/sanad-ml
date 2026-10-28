"""Serve the Sanad visualizer at http://localhost:8000 (works with no internet).

Run from the repo root:  uv run python -I tools/visualizer/serve.py
Web Serial (USB) needs a secure context, and http://localhost counts as one.
"""

import argparse
import functools
import http.server
import webbrowser
from pathlib import Path
from typing import ClassVar

HERE = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
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

    def log_request(self, code="-", size="-"):
        if isinstance(code, int) and code >= 400:
            super().log_request(code, size)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args()

    handler = functools.partial(Handler, directory=str(HERE))
    for port in range(args.port, args.port + 10):
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
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
