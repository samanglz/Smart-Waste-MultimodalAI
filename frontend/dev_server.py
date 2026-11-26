from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parent
API_BASE = os.environ.get("SMART_WASTE_API_URL", "http://127.0.0.1:8000")

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def proxy(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else None

        headers = {}
        for name in ("Content-Type", "Accept"):
            value = self.headers.get(name)
            if value:
                headers[name] = value

        request = Request(
            API_BASE + self.path,
            data=body,
            headers=headers,
            method=self.command,
        )

        try:
            response = urlopen(request, timeout=300)
            status = response.status
            payload = response.read()
            content_type = response.headers.get("Content-Type")
        except HTTPError as exc:
            status = exc.code
            payload = exc.read()
            content_type = exc.headers.get("Content-Type")
        except URLError as exc:
            status = 502
            payload = json.dumps({
                "detail": f"Could not reach API server: {exc.reason}"
            }).encode("utf-8")
            content_type = "application/json"

        self.send_response(status)
        if content_type:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()

        if payload:
            self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith("/api/") or self.path == "/health":
            self.proxy()
        else:
            super().do_GET()

    def do_POST(self):
        self.proxy()

    def do_DELETE(self):
        self.proxy()

if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8080), Handler)
    print("Frontend: http://127.0.0.1:8080")
    print(f"API: {API_BASE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFrontend server stopped.")
    finally:
        server.server_close()