"""Measure retrieval precision/recall of POST /api/search against datasets/synthetic/test/retrieval_queries.json.

    (backend running)  python datasets/scripts/eval_retrieval.py
Replace the printed numbers into MEASURED in frontend/index.html to show real results on the Analytics page.
"""
import json, sys, urllib.request
from pathlib import Path

API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
Q = json.loads((Path(__file__).resolve().parents[1] / "synthetic/test/retrieval_queries.json").read_text())
tp = fp = fn = 0
for q in Q:
    req = urllib.request.Request(API + "/api/search", data=json.dumps(q["filters"]).encode(), headers={"Content-Type": "application/json"})
    got = {e["event_id"] for e in json.load(urllib.request.urlopen(req))["events"]}
    rel = set(q["relevant_event_ids"])
    tp += len(got & rel); fp += len(got - rel); fn += len(rel - got)
    print(f"{len(got & rel):>2}/{len(rel):<2} relevant found  {len(got - rel)} extra   {q['query']}")
p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
print(f"\nprecision={p:.3f} recall={r:.3f}")
