# VisionGuard AI — project folder

Multi-stream video intelligence: conversational search, evidence, clarify-once memory, cross-camera tracking, alerts.

```
visionguard-ai/
├── frontend/index.html            React UI (single file, loads React from CDN — needs internet)
├── backend/
│   ├── app/main.py                FastAPI: REST + WebSocket (/ws/alerts) + serves the frontend
│   ├── app/db.py                  Database connection (SQLAlchemy; SQLite default, PostgreSQL optional)
│   ├── app/seed_data.json         Demo cameras / events / alerts / memory loaded on first start
│   ├── pipeline/process_video.py  OPTIONAL YOLO + ByteTrack -> API (untested, see below)
│   └── data/visionguard.db        created automatically on first run
├── datasets/
│   ├── synthetic/train/           detections.csv (6000 rows), reid_pairs.csv (4000 rows)
│   ├── synthetic/test/            detections.csv (1500 rows), reid_pairs.csv (1000 rows), retrieval_queries.json
│   ├── scripts/generate_synthetic.py   regenerate the data
│   ├── scripts/train_models.py         train + evaluate alert classifier and re-ID linker
│   ├── scripts/eval_retrieval.py       precision/recall of /api/search against test queries
│   └── scripts/download_real_datasets.md   real public datasets (UA-DETRAC, MOT, CityFlow, VIRAT, UCF-Crime, COCO)
├── scripts/run.sh | run.bat       start the server
└── requirements.txt
```

## Run it (VS Code)
1. Install Python 3.10+. Open this folder in VS Code (File > Open Folder).
2. Terminal > New Terminal, then:
   ```
   python -m venv .venv
   # Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn backend.app.main:app --reload --port 8000
   ```
   (or run `scripts/run.sh` / `scripts/run.bat`)
3. Open http://localhost:8000 — the top bar shows **Live backend** and data is read from / written to the database.
   API docs: http://localhost:8000/docs
4. No backend? Open `frontend/index.html` directly: it falls back to built-in mock data (top bar shows **Mock data**).
   If you serve the frontend separately (e.g. Live Server on :5500) the page also tries http://localhost:8000.

## Database
Default is SQLite (`backend/data/visionguard.db`, created and seeded on first start; delete the file to reset).
PostgreSQL: `pip install psycopg2-binary`, then
`set DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/visionguard` (Windows) /
`export DATABASE_URL=...` (macOS/Linux) before starting. Tables: cameras, events, detections, alerts, memory, standing_queries, tracks.

## Datasets — read this
The datasets here are **synthetic and structured** (no video): rows are generated from rules with ~3% label noise,
so models score very high (alert classifier ~0.97 accuracy, re-ID linker ~0.99 on the test split). They let you exercise
training/testing code and the retrieval evaluation. They are NOT evidence of real-world accuracy.
Variables: `detections.csv` features = hour, confidence, dwell_sec, speed_px_s, in_restricted_zone, persons_in_frame, box size,
object_type, camera; labels = `event_type`, `is_alert`. `reid_pairs.csv` features = time_gap_s, camera_hops, color_match,
size_ratio, embedding_cosine; label = `same_identity`. For real results download the public datasets listed in
`datasets/scripts/download_real_datasets.md` (large, licence terms apply).

Commands:
```
python datasets/scripts/generate_synthetic.py
pip install scikit-learn numpy && python datasets/scripts/train_models.py
python datasets/scripts/eval_retrieval.py          # backend must be running
```

## What is real and what is not
- Tested: database seeding, all REST endpoints, WebSocket alert broadcast, frontend in live and mock modes, retrieval evaluation (1.00 on the 8 seed queries).
- Text-to-filter parsing (natural language -> object/colour/place/time) runs in the browser with rules, not an LLM. `/api/search` takes structured filters; plug an LLM or CLIP text encoder in front of it.
- The camera pictures in the UI are drawn placeholder scenes. Nothing analyses real video until you run `backend/pipeline/process_video.py`
  (YOLO + ByteTrack; **not tested here**; needs `pip install ultralytics opencv-python requests` and a video). Zone logic, CLIP/SigLIP re-ID embeddings, FAISS/Qdrant and a real alert engine are not implemented.
- Analytics numbers are placeholders (`MEASURED` in frontend/index.html). Replace them with output of `eval_retrieval.py`.
