"""Generate SYNTHETIC structured datasets (no video) for training/testing the VisionGuard logic.

    python datasets/scripts/generate_synthetic.py            # writes datasets/synthetic/{train,test}/*.csv

Outputs
  train|test/detections.csv   per-detection features + labels (event_type, is_alert)
  train|test/reid_pairs.csv   candidate track pairs + label same_identity  (cross-camera re-ID linker)
  test/retrieval_queries.json structured queries with ground-truth event_ids from the backend seed events

These are rule-generated with ~3% label noise, so models reach very high scores. They exist to exercise the
pipeline end to end. For real results use the public datasets listed in datasets/README.md.
"""
import csv, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "synthetic"
SEED = Path(__file__).resolve().parents[2] / "backend" / "app" / "seed_data.json"
random.seed(42)
CAMS = [f"CAM-0{i}" for i in range(1, 9)]
OBJ = ["person", "car", "truck", "bike", "bag"]
COL = ["red", "blue", "black", "white", "grey"]
DET_COLS = ["detection_id", "camera_id", "timestamp_sec", "hour", "object_type", "color", "track_id", "x", "y", "w", "h",
            "confidence", "dwell_sec", "speed_px_s", "in_restricted_zone", "persons_in_frame", "event_type", "is_alert"]


def make_det(i):
    obj = random.choices(OBJ, [.5, .25, .07, .06, .12])[0]
    cam = random.choice(CAMS)
    hour = random.choices(range(24), [1]*6 + [3]*3 + [4]*9 + [3]*3 + [1]*3)[0]
    ts = hour * 3600 + random.randint(0, 3599)
    w = random.randint(50, 90) if obj == "person" else random.randint(40, 70) if obj == "bag" else random.randint(120, 220)
    h = int(w * (2.4 if obj == "person" else .8 if obj == "bag" else .66))
    x, y = random.randint(0, 640 - w), random.randint(20, 360 - h)
    conf = round(min(99.5, max(55, random.gauss(89, 7))), 1)
    dwell = round(random.expovariate(1 / 25), 1)
    if random.random() < .06: dwell += random.uniform(90, 400)
    speed = round(abs(random.gauss(30 if obj != "person" else 12, 10)), 1)
    zone = int(cam == "CAM-06" and x > 290 and x + w < 600 and random.random() < .55)
    persons = max(0, int(random.gauss(2, 1.6))) + (random.randint(4, 7) if random.random() < .04 else 0)
    night = hour >= 20 or hour < 6
    if zone and obj == "person" and night and conf > 85: ev = "Restricted Zone Entry"
    elif obj == "person" and dwell > 120: ev = "Loitering"
    elif obj == "bag" and dwell > 60 and persons < 2: ev = "Object Left"
    elif persons >= 5: ev = "Crowd"
    elif obj == "person" and speed > 28: ev = "Running"
    elif obj in ("car", "truck", "bike") and cam in ("CAM-01", "CAM-08"): ev = "Entry" if random.random() < .5 else "Exit"
    else: ev = "Person detected" if obj == "person" else "Vehicle detected" if obj != "bag" else "Object detected"
    alert = int(ev in ("Restricted Zone Entry", "Loitering", "Object Left", "Crowd"))
    if random.random() < .03: alert = 1 - alert  # label noise
    return [f"D-{i:06d}", cam, ts, hour, obj, random.choice(COL) if obj != "person" else random.choice(["black", "blue", "grey", "red"]),
            f"T-{random.randint(1, 900)}", x, y, w, h, conf, round(dwell, 1), speed, zone, persons, ev, alert]


def make_pair(i):
    same = random.random() < .5
    col_a = random.choice(COL)
    col_b = col_a if (same and random.random() < .93) or (not same and random.random() < .2) else random.choice(COL)
    gap = random.randint(20, 400) if same else random.randint(5, 3000)
    hop = random.choice([1, 1, 2, 2, 3]) if same else random.randint(1, 7)
    size = round(random.gauss(1.0, .08 if same else .35), 3)
    emb = round(random.gauss(.82 if same else .35, .08 if same else .17), 3)  # cosine sim of appearance embeddings
    return [f"P-{i:06d}", gap, hop, int(col_a == col_b), size, emb, int(same)]


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def retrieval_queries():
    seed = json.loads(SEED.read_text())
    ev = seed["EVENTS0"]
    veh = {"car", "truck", "bike", "van"}
    defs = [
        ("Did a red car enter the main gate?", dict(object_type="vehicle", color="red", camera_id="CAM-01", event_type="Entry")),
        ("Find a person carrying a large black bag.", dict(object_type="person", tag="bag")),
        ("Show people entering the restricted zone.", dict(object_type="person", zone="Restricted Zone")),
        ("Show all vehicles that entered.", dict(object_type="vehicle", event_type="Entry")),
        ("Anyone loitering?", dict(event_type="Loitering")),
        ("Any bag left behind?", dict(event_type="Object Left")),
        ("Black vehicles that exited.", dict(object_type="vehicle", color="black", event_type="Exit")),
        ("What did CAM-02 see?", dict(camera_id="CAM-02")),
    ]
    def ok(e, f):
        if f.get("object_type") == "vehicle" and e["object_type"] not in veh: return False
        if f.get("object_type") not in (None, "vehicle") and e["object_type"] != f["object_type"]: return False
        for k in ("color", "camera_id", "event_type", "zone"):
            if k in f and e.get(k) != f[k]: return False
        if "tag" in f and f["tag"] not in (e.get("tags") or []): return False
        return True
    return [{"query": q, "filters": f, "relevant_event_ids": [e["event_id"] for e in ev if ok(e, f)]} for q, f in defs]


if __name__ == "__main__":
    N_TRAIN, N_TEST = 6000, 1500
    write(ROOT / "train" / "detections.csv", DET_COLS, [make_det(i) for i in range(N_TRAIN)])
    write(ROOT / "test" / "detections.csv", DET_COLS, [make_det(100000 + i) for i in range(N_TEST)])
    PC = ["pair_id", "time_gap_s", "camera_hops", "color_match", "size_ratio", "embedding_cosine", "same_identity"]
    write(ROOT / "train" / "reid_pairs.csv", PC, [make_pair(i) for i in range(4000)])
    write(ROOT / "test" / "reid_pairs.csv", PC, [make_pair(100000 + i) for i in range(1000)])
    (ROOT / "test" / "retrieval_queries.json").write_text(json.dumps(retrieval_queries(), indent=1))
    print("wrote", N_TRAIN, "train /", N_TEST, "test detections; 4000/1000 re-ID pairs; retrieval queries")
