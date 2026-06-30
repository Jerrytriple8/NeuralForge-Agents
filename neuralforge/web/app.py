"""FastAPI monitoring dashboard for NeuralForge pipelines."""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError:
    FastAPI = None  # type: ignore[assignment,misc]

from neuralforge.observability.metrics import MetricsCollector


# HTML dashboard template
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuralForge Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0e17; color: #e0e6ed; }
        .header { background: linear-gradient(135deg, #1a1f35, #0d1220); padding: 20px 30px;
                   border-bottom: 1px solid #2a3050; display: flex; align-items: center; gap: 15px; }
        .header h1 { font-size: 24px; color: #6c7ee1; }
        .header .status { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .status.running { background: #1a3d2a; color: #4ade80; }
        .status.idle { background: #3d3a1a; color: #facc15; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; padding: 20px; }
        .card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 20px; }
        .card h3 { color: #818cf8; margin-bottom: 12px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
        .metric { font-size: 36px; font-weight: 700; color: #f0f4ff; }
        .metric-label { font-size: 12px; color: #6b7280; margin-top: 4px; }
        .log-entry { padding: 8px 0; border-bottom: 1px solid #1f2937; font-family: monospace; font-size: 13px; }
        .log-entry .ts { color: #6b7280; }
        .log-entry .level-INFO { color: #60a5fa; }
        .log-entry .level-WARN { color: #fbbf24; }
        .log-entry .level-ERROR { color: #f87171; }
        #ws-status { position: fixed; bottom: 20px; right: 20px; padding: 8px 16px;
                     border-radius: 20px; font-size: 12px; }
        .ws-connected { background: #1a3d2a; color: #4ade80; }
        .ws-disconnected { background: #3d1a1a; color: #f87171; }
    </style>
</head>
<body>
    <div class="header">
        <h1>NeuralForge</h1>
        <span class="status running" id="pipeline-status">IDLE</span>
    </div>
    <div class="grid">
        <div class="card">
            <h3>Pipelines Executed</h3>
            <div class="metric" id="metric-pipelines">0</div>
        </div>
        <div class="card">
            <h3>Nodes Completed</h3>
            <div class="metric" id="metric-nodes">0</div>
        </div>
        <div class="card">
            <h3>Avg Duration</h3>
            <div class="metric" id="metric-duration">0ms</div>
        </div>
        <div class="card">
            <h3>Error Rate</h3>
            <div class="metric" id="metric-errors">0%</div>
        </div>
        <div class="card" style="grid-column: span 2;">
            <h3>Live Logs</h3>
            <div id="logs" style="max-height: 300px; overflow-y: auto;"></div>
        </div>
    </div>
    <div id="ws-status" class="ws-disconnected">DISCONNECTED</div>
    <script>
        const ws = new WebSocket(`ws://${location.host}/ws/pipeline`);
        const wsStatus = document.getElementById('ws-status');
        const logs = document.getElementById('logs');
        ws.onopen = () => { wsStatus.className = 'ws-connected'; wsStatus.textContent = 'LIVE'; };
        ws.onclose = () => { wsStatus.className = 'ws-disconnected'; wsStatus.textContent = 'DISCONNECTED'; };
        ws.onmessage = (e) => {
            const data = JSON.parse(e.data);
            if (data.type === 'metrics') {
                document.getElementById('metric-pipelines').textContent = data.pipelines || 0;
                document.getElementById('metric-nodes').textContent = data.nodes || 0;
                document.getElementById('metric-duration').textContent = (data.avg_duration || 0) + 'ms';
                document.getElementById('metric-errors').textContent = (data.error_rate || 0) + '%';
            }
            if (data.type === 'log') {
                const div = document.createElement('div');
                div.className = 'log-entry';
                div.innerHTML = `<span class="ts">${new Date(data.ts*1000).toLocaleTimeString()}</span> <span class="level-${data.level}">${data.level}</span> ${data.message}`;
                logs.prepend(div);
            }
        };
    </script>
</body>
</html>"""


def create_dashboard_app(metrics: MetricsCollector | None = None) -> Any:
    """Create and configure the FastAPI dashboard application.

    Args:
        metrics: Shared MetricsCollector instance.

    Returns:
        FastAPI application instance.
    """
    if FastAPI is None:
        raise ImportError("FastAPI is required: pip install fastapi uvicorn")

    app = FastAPI(title="NeuralForge Dashboard", version="0.1.0")
    _metrics = metrics or MetricsCollector()
    _ws_clients: list[Any] = []

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return _DASHBOARD_HTML

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/metrics")
    async def get_metrics():
        return _metrics.export()

    @app.get("/api/pipelines")
    async def list_pipelines():
        return {"pipelines": [], "message": "Pipeline history endpoint"}

    @app.websocket("/ws/pipeline")
    async def websocket_pipeline(ws: WebSocket):
        await ws.accept()
        _ws_clients.append(ws)
        try:
            while True:
                data = await ws.receive_text()
                # Client can send commands
        except WebSocketDisconnect:
            _ws_clients.remove(ws)

    # Broadcast helper
    async def broadcast(message: dict[str, Any]) -> None:
        dead = []
        for ws in _ws_clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.remove(ws)

    app.broadcast = broadcast  # type: ignore[attr-defined]
    return app
