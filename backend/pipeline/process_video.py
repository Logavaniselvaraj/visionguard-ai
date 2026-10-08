"""OPTIONAL real-video pipeline: YOLO detection + ByteTrack tracking -> VisionGuard API.

NOT TESTED in the delivered project (needs a GPU/CPU, a video file and the packages below).
    pip install ultralytics opencv-python requests
    python backend/pipeline/process_video.py --source path/to/video.mp4 --camera CAM-01

What it does: runs YOLO with ByteTrack, estimates a dominant colour per tracked object,
turns first-appearance of each track into an "Entry"/"Vehicle detected" event and posts
detections + events to the running backend. Zone logic, re-identification embeddings
(CLIP/SigLIP + FAISS/Qdrant) and the alert engine are the next steps.
"""
import argparse
import time

import cv2
import requests
from ultralytics import YOLO

NAMES = {0: "person", 2: "car", 3: "bike", 5: "truck", 7: "truck", 24: "bag", 26: "bag", 28: "bag"}


def dominant_color(crop):
    if crop.size == 0:
        return "grey"
    h, s, v = cv2.cvtColor(cv2.resize(crop, (16, 16)), cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(0)
    if v < 60: return "black"
    if s < 40: return "white" if v > 170 else "grey"
    if h < 10 or h > 170: return "red"
    if 100 < h < 130: return "blue"
    return "grey"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--camera", default="CAM-01")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--model", default="yolov8n.pt")
    a = ap.parse_args()
    model, seen = YOLO(a.model), set()
    for r in model.track(source=a.source, stream=True, persist=True, tracker="bytetrack.yaml", verbose=False):
        now = time.localtime()
        t = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
        dets = []
        for b in (r.boxes if r.boxes is not None else []):
            cls = int(b.cls[0])
            if cls not in NAMES or b.id is None:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            tid, conf, obj = f"T-{int(b.id[0])}", float(b.conf[0]) * 100, NAMES[cls]
            color = dominant_color(r.orig_img[y1:y2, x1:x2])
            dets.append({"detection_id": f"{a.camera}-{t}-{tid}", "camera_id": a.camera, "timestamp": time.strftime("%H:%M:%S"),
                         "object_type": obj, "track_id": tid, "bounding_box": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                         "confidence": conf, "embedding": None})
            if tid not in seen:
                seen.add(tid)
                requests.post(f"{a.api}/api/events", json={
                    "event_id": f"EV-{a.camera}-{t}-{tid}", "camera_id": a.camera, "t": t, "ms": 0,
                    "event_type": "Vehicle detected" if obj in ("car", "truck", "bike") else "Person detected",
                    "zone": None, "object_type": obj, "color": color, "confidence": round(conf, 1),
                    "label": f"{color.title()} {obj.title()}", "tags": [], "track_id": tid,
                    "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}, "evidence_url": None})
        if dets:
            requests.post(f"{a.api}/api/detections", json=dets)


if __name__ == "__main__":
    main()
