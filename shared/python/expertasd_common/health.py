"""Tiny stdlib health endpoint shared by all workers.

Serves GET /health on a background thread. The worker flips the state
from "loading" to "ready" only after the model is resident in memory,
so container healthchecks reflect actual readiness, not process start.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


class HealthState:
    def __init__(self, model: str):
        self._lock = threading.Lock()
        self._status = "loading"
        self._detail: Optional[str] = None
        self.model = model

    def set(self, status: str, detail: Optional[str] = None) -> None:
        with self._lock:
            self._status = status
            self._detail = detail

    def snapshot(self) -> dict:
        with self._lock:
            return {"status": self._status, "model": self.model, "detail": self._detail}


def start_health_server(state: HealthState, port: int = 8080) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            snap = state.snapshot()
            code = 200 if snap["status"] == "ready" else 503
            body = json.dumps(snap).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence per-probe access logs
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
