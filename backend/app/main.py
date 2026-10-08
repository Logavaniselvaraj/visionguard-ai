"""VisionGuard AI — FastAPI backend.

Run:  uvicorn backend.app.main:app --reload --port 8000   (from the project root)
Docs: http://localhost:8000/docs
The frontend in /frontend is served at http://localhost:8000/
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import delete, select, update

from . import db
from .db import alerts, cameras, detections, engine, events, memory, rows, standing_queries, tracks

app = FastAPI(title="VisionGuard AI API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    db.init_db()


# ---------------- WebSocket hub for real-time alerts ----------------
class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def broadcast(self, payload: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(ws)
        for d in dead:
            self.clients.discard(d)


hub = Hub()


@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive; clients do not send data
    except WebSocketDisconnect:
        hub.clients.discard(ws)


# ---------------- models ----------------
class AlertIn(BaseModel):
    alert: dict
    event: dict


class SearchIn(BaseModel):
    object_type: Optional[str] = None      # person | car | truck | bike | bag | vehicle
    color: Optional[str] = None
    camera_id: Optional[str] = None
    event_type: Optional[str] = None
    zone: Optional[str] = None
    tag: Optional[str] = None
    t_from: Optional[int] = None           # seconds since midnight
    t_to: Optional[int] = None
    limit: int = 50


class MemoryIn(BaseModel):
    memory_id: str
    key: str
    value: str
    camera_id: str
    zone: Optional[str] = None
    created_at: str


class StatusIn(BaseModel):
    status: str


# ---------------- REST ----------------
@app.get("/api/health")
def health():
    return {"status": "ok", "db": str(engine.url).split("@")[-1], "time": time.time()}


@app.get("/api/cameras")
def get_cameras():
    with engine.connect() as c:
        return rows(c, select(cameras).order_by(cameras.c.camera_id))


@app.get("/api/events")
def get_events():
    with engine.connect() as c:
        return rows(c, select(events).order_by(events.c.t))


@app.post("/api/search")
def search(q: SearchIn):
    """Structured retrieval over indexed events. The NLU step (text -> filters) lives in the
    frontend today; replace it with an LLM / CLIP text encoder and call this endpoint."""
    with engine.connect() as c:
        data = rows(c, select(events).order_by(events.c.confidence.desc()))
    veh = {"car", "truck", "bike", "van"}

    def ok(e):
        if q.object_type == "vehicle" and e["object_type"] not in veh: return False
        if q.object_type and q.object_type != "vehicle" and e["object_type"] != q.object_type: return False
        if q.color and e["color"] != q.color: return False
        if q.camera_id and e["camera_id"] != q.camera_id: return False
        if q.event_type and e["event_type"] != q.event_type: return False
        if q.zone and e["zone"] != q.zone: return False
        if q.tag and q.tag not in (e["tags"] or []): return False
        if q.t_from is not None and e["t"] < q.t_from: return False
        if q.t_to is not None and e["t"] > q.t_to: return False
        return True

    res = [e for e in data if ok(e)][: q.limit]
    return {"count": len(res), "events": res}


@app.get("/api/tracks/{track_id}")
def get_track(track_id: str):
    with engine.connect() as c:
        t = rows(c, select(tracks).where(tracks.c.id == track_id))
        ev = rows(c, select(events).where(events.c.track_id == track_id).order_by(events.c.t))
    if not t and not ev:
        raise HTTPException(404, "Unknown track")
    return {"track": t[0] if t else {"id": track_id}, "events": ev}


@app.get("/api/alerts")
def get_alerts():
    with engine.connect() as c:
        return rows(c, select(alerts).order_by(alerts.c.created_at.desc(), alerts.c.t.desc()))


@app.post("/api/alerts")
async def post_alert(body: AlertIn):
    a, e = body.alert, body.event
    with engine.begin() as c:
        c.execute(events.insert().prefix_with("OR IGNORE") if str(engine.url).startswith("sqlite") else events.insert(),
                  {k: v for k, v in e.items() if k in events.c})
        c.execute(alerts.insert(), {k: v for k, v in {**a, "created_at": a.get("createdAt", time.time() * 1000)}.items() if k in alerts.c})
    await hub.broadcast({"alert": a, "event": e})
    return {"ok": True}


@app.patch("/api/alerts/{alert_id}")
def patch_alert(alert_id: str, body: StatusIn):
    with engine.begin() as c:
        c.execute(update(alerts).where(alerts.c.alert_id == alert_id).values(status=body.status))
    return {"ok": True}


@app.get("/api/memory")
def get_memory():
    with engine.connect() as c:
        return rows(c, select(memory).order_by(memory.c.created_at))


@app.put("/api/memory")
def put_memory(items: list[MemoryIn]):
    """Replace the whole memory table (simple sync used by the UI)."""
    with engine.begin() as c:
        c.execute(delete(memory))
        if items:
            c.execute(memory.insert(), [i.model_dump() for i in items])
    return {"ok": True, "count": len(items)}


@app.get("/api/standing-queries")
def get_sq():
    with engine.connect() as c:
        return [r["body"] for r in rows(c, select(standing_queries))]


@app.put("/api/standing-queries")
def put_sq(items: list[dict[str, Any]]):
    with engine.begin() as c:
        c.execute(delete(standing_queries))
        if items:
            c.execute(standing_queries.insert(), [{"sq_id": i.get("sq_id", str(n)), "body": i} for n, i in enumerate(items)])
    return {"ok": True}


@app.post("/api/detections")
def ingest_detections(items: list[dict[str, Any]]):
    """Endpoint the video pipeline (backend/pipeline/process_video.py) posts to."""
    with engine.begin() as c:
        for i in items:
            c.execute(detections.insert(), {k: v for k, v in i.items() if k in detections.c})
    return {"ok": True, "count": len(items)}


@app.post("/api/events")
async def ingest_event(body: dict[str, Any]):
    with engine.begin() as c:
        c.execute(events.insert(), {k: v for k, v in body.items() if k in events.c})
    return {"ok": True}


# ---------------- static frontend ----------------
FRONT = Path(__file__).resolve().parents[2] / "frontend"
if FRONT.exists():
    app.mount("/", StaticFiles(directory=FRONT, html=True), name="frontend")
