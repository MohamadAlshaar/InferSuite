from __future__ import annotations

import json
import threading
from collections import deque
from typing import Any, Dict, List

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()

_LOCK = threading.Lock()
_TRACE_HISTORY: deque[Dict[str, Any]] = deque(maxlen=200)
_LATEST_TRACE: Dict[str, Any] | None = None


def record_trace(trace: Dict[str, Any]) -> None:
    global _LATEST_TRACE
    with _LOCK:
        _LATEST_TRACE = trace
        _TRACE_HISTORY.append(trace)


@router.get("/viz/latest")
async def viz_latest():
    with _LOCK:
        return JSONResponse(_LATEST_TRACE or {})


@router.get("/viz/history")
async def viz_history(limit: int = Query(default=50, ge=1, le=200)):
    with _LOCK:
        items: List[Dict[str, Any]] = list(_TRACE_HISTORY)[-limit:]
    return JSONResponse(items)


@router.get("/viz", response_class=HTMLResponse)
async def viz_page():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>LLM Service Viz</title>
  <style>
    body { font-family: monospace; margin: 24px; background: #111; color: #eee; }
    h1 { font-size: 20px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { background: #1b1b1b; padding: 16px; border-radius: 12px; border: 1px solid #333; }
    pre { white-space: pre-wrap; word-break: break-word; }
    .muted { color: #aaa; }
  </style>
</head>
<body>
  <h1>LLM Service Viz</h1>
  <p class="muted">Auto-refreshes every 2 seconds.</p>
  <div class="grid">
    <div class="card">
      <h2>Latest Trace</h2>
      <pre id="latest">loading...</pre>
    </div>
    <div class="card">
      <h2>Recent History</h2>
      <pre id="history">loading...</pre>
    </div>
  </div>
<script>
async function refresh() {
  const latest = await fetch('/viz/latest').then(r => r.json()).catch(() => ({}));
  const history = await fetch('/viz/history?limit=10').then(r => r.json()).catch(() => ([]));
  document.getElementById('latest').textContent = JSON.stringify(latest, null, 2);
  document.getElementById('history').textContent = JSON.stringify(history, null, 2);
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""
    return HTMLResponse(html)
