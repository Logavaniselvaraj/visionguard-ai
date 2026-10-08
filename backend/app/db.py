"""Database connection layer (SQLAlchemy Core).

Default: SQLite file at backend/data/visionguard.db
PostgreSQL: set DATABASE_URL, e.g.
    postgresql+psycopg2://user:password@localhost:5432/visionguard
"""
import json
import os
from pathlib import Path

from sqlalchemy import (JSON, Column, Float, Integer, MetaData, String, Table,
                        create_engine, select)

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'visionguard.db'}")

engine = create_engine(
    DATABASE_URL,
    future=True,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
md = MetaData()

cameras = Table("cameras", md,
    Column("camera_id", String(16), primary_key=True), Column("name", String(80)),
    Column("location", String(120)), Column("scene", String(24)), Column("status", String(16), default="online"),
    Column("fps", Integer, default=25), Column("last_update", String(32), default="live"))

events = Table("events", md,
    Column("event_id", String(24), primary_key=True), Column("camera_id", String(16), index=True),
    Column("t", Integer, index=True),            # seconds since midnight
    Column("ms", Integer, default=0), Column("event_type", String(40)), Column("zone", String(60)),
    Column("object_type", String(16)), Column("color", String(16)), Column("confidence", Float),
    Column("label", String(80)), Column("tags", JSON), Column("track_id", String(24), index=True),
    Column("bbox", JSON), Column("evidence_url", String(255)))

detections = Table("detections", md,
    Column("detection_id", String(32), primary_key=True), Column("camera_id", String(16), index=True),
    Column("timestamp", String(32)), Column("object_type", String(16)), Column("track_id", String(24)),
    Column("bounding_box", JSON), Column("confidence", Float), Column("embedding", JSON))

alerts = Table("alerts", md,
    Column("alert_id", String(16), primary_key=True), Column("rule", String(120)), Column("camera_id", String(16)),
    Column("t", Integer), Column("severity", String(12)), Column("confidence", Float), Column("status", String(16)),
    Column("day", String(12), default="today"), Column("event_id", String(24)),
    Column("created_at", Float, default=0), Column("source", String(200)))

memory = Table("memory", md,
    Column("memory_id", String(24), primary_key=True), Column("key", String(80)), Column("value", String(80)),
    Column("camera_id", String(16)), Column("zone", String(60)), Column("created_at", String(24)))

standing_queries = Table("standing_queries", md,
    Column("sq_id", String(24), primary_key=True), Column("body", JSON))

tracks = Table("tracks", md,
    Column("id", String(24), primary_key=True), Column("label", String(80)), Column("kind", String(16)), Column("conf", Float))


def init_db(seed: bool = True):
    md.create_all(engine)
    if not seed:
        return
    with engine.begin() as c:
        if c.execute(select(cameras.c.camera_id).limit(1)).first():
            return  # already seeded
        S = json.loads((BASE / "seed_data.json").read_text())
        c.execute(cameras.insert(), [{k: v for k, v in x.items() if k in cameras.c} for x in S["CAMS0"]])
        c.execute(events.insert(), [{k: v for k, v in x.items() if k in events.c} for x in S["EVENTS0"]])
        c.execute(alerts.insert(), [{**{k: v for k, v in x.items() if k in alerts.c}, "created_at": 0} for x in S["ALERTS0"]])
        c.execute(memory.insert(), [{k: v for k, v in x.items() if k in memory.c} for x in S["MEM0"]])
        c.execute(tracks.insert(), [
            {"id": "CAR-008", "label": "Red Sedan", "kind": "vehicle", "conf": 91},
            {"id": "PER-114", "label": "Person with black bag", "kind": "person", "conf": 88}])


def rows(conn, stmt):
    return [dict(r._mapping) for r in conn.execute(stmt)]
