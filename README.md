# Sports Performance Analytics Platform

An end-to-end football analytics system combining ensemble machine learning for player performance prediction with YOLOv8 computer vision for match video analysis. Built as a full-stack application with a FastAPI backend and Streamlit frontend.

---

## Problem Statement

Professional football clubs generate terabytes of match footage and player statistics annually but lack accessible tools to extract actionable intelligence without expensive proprietary software. This platform solves two distinct problems:

1. **Scouting and Valuation** — Given a player's physical and performance attributes, predict their Overall rating and market value in EUR using ensemble ML models trained on real footballer statistics.
2. **Match Analytics** — Given a broadcast video clip, automatically detect all visible players, assign persistent identities across frames, and compute per-player speed and distance metrics without manual annotation.

---

## Real-World Applications

- Club scouting departments valuing transfer targets against statistical benchmarks
- Youth academy performance tracking using objective physical attribute scoring
- Sports science staff monitoring physical load from match broadcast footage
- Academic research into applied computer vision and sports analytics

---

## Features

| Module | Capability |
|---|---|
| Player Performance Prediction | Predict Overall rating and market value (EUR) from 6 physical attributes |
| Injury Risk Assessment | Classify risk as Low / Medium / High using a weighted age-stamina-workrate formula |
| Video Player Tracking | YOLOv8n + ByteTracker detection with persistent player ID assignment |
| Speed Estimation | Per-player average and top speed (km/h) from pixel displacement and frame timing |
| Distance Estimation | Cumulative distance covered per player across the visible clip (metres) |
| Async Job Queue | Non-blocking video processing — upload returns a job ID immediately, poll for progress |
| Annotated Video Export | Download output MP4 with bounding boxes and track IDs overlaid per frame |

---

## Tech Stack

| Layer | Technology | Detail |
|---|---|---|
| Backend API | FastAPI (Python 3.10) | REST API, async background tasks, model serving |
| Frontend UI | Streamlit | Interactive analytics dashboard, job polling client |
| Object Detection | YOLOv8n (Ultralytics) | Person class only (class=0), conf threshold 0.60 |
| Multi-Object Tracking | ByteTracker (Ultralytics built-in) | Persistent player ID across frames |
| Player Rating Model | XGBRegressor | n_estimators=200, learning_rate=0.05, max_depth=6 |
| Market Value Model | RandomForestRegressor | n_estimators=200 |
| Video I/O | OpenCV (cv2) | Frame extraction, bounding box rendering, MP4 writing |
| Model Persistence | joblib | Serialize trained sklearn and XGBoost models |
| Data Processing | pandas, numpy | CSV ingestion, currency string parsing, feature engineering |
| Containerization | Docker (python:3.10-slim + ffmpeg) | Reproducible single-stage backend image |

---

## System Architecture

```
  ┌──────────────────────────────────────────┐
  │  frontend.py  (Streamlit — port 8501)    │
  │  - Player Prediction module              │
  │  - Injury Risk module                    │
  │  - Video Tracking module (job polling)   │
  └──────────────────┬───────────────────────┘
                     │ HTTP (requests library)
                     ▼
  ┌──────────────────────────────────────────┐
  │  app/main.py  (FastAPI — port 8001)      │
  │  - GET  /                                │
  │  - POST /predict                         │
  │  - POST /injury-risk                     │
  │  - POST /process-video  (HTTP 202)       │
  │  - GET  /job-status/{job_id}             │
  │  - GET  /download-video/{job_id}         │
  │                                          │
  │  In-process job store (dict + Lock)      │
  │  BackgroundTasks thread per upload       │
  └──────────────────┬───────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────┐
  │  app/analyzer.py  (SportsAnalyzer)       │
  │                                          │
  │  ML pipeline:                            │
  │    Footballer.csv → preprocess           │
  │    → XGBRegressor (Overall)              │
  │    → RandomForestRegressor (Value)       │
  │    → joblib serialize → models/*.pkl     │
  │                                          │
  │  Video pipeline:                         │
  │    MP4 → cv2.VideoCapture                │
  │    → sample every 3rd frame              │
  │    → YOLO.track() class=0 conf=0.60     │
  │    → zone filter (top 15%, bot 8%)       │
  │    → box-height filter (5–28% of h)      │
  │    → ByteTracker ID assignment           │
  │    → track_history accumulate            │
  │    → annotate + write frame              │
  │    → filter stubs (min 20 frames)        │
  │    → compute speed / distance analytics  │
  │    → output MP4 + analytics dict         │
  └──────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10
- pip
- `data/Footballer.csv` (FIFA footballer statistics dataset, not committed to git)

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd sports-analytics

# 2. Create virtual environment
python -m venv myvenv

# Windows (bash / Git Bash)
source myvenv/Scripts/activate

# Windows (cmd)
myvenv\Scripts\activate.bat

# macOS / Linux
source myvenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create required directories and place dataset
mkdir -p data models output
# Copy Footballer.csv into data/
```

Models are auto-generated on first backend start — no separate training step required.

### Docker (backend only)

```bash
docker build -t sports-analytics .
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  sports-analytics
```

The Docker image exposes port **8000**. The Streamlit frontend must run separately on the host.

---

## Usage

Both services must run simultaneously in separate terminals.

```bash
# Terminal 1 — start backend first
uvicorn app.main:app --reload --port 8001

# Terminal 2 — start frontend
streamlit run frontend.py
```

Open `http://localhost:8501` in your browser. The sidebar selects the active module.

On first run with no pre-trained models, the backend trains both models from `data/Footballer.csv` before serving requests (approximately 60–120 seconds — logged to console).

### Testing Endpoints Directly

```bash
# Health check
curl http://localhost:8001/

# Player prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"age":24,"potential":85,"stamina":80,"strength":75,"sprint_speed":90,"work_rate_encoded":2}'

# Injury risk
curl -X POST http://localhost:8001/injury-risk \
  -H "Content-Type: application/json" \
  -d '{"age":28,"stamina":70,"work_rate":3}'

# Upload video
curl -X POST http://localhost:8001/process-video \
  -F "file=@match_clip.mp4"
# Returns: {"job_id": "<uuid>", "status": "pending"}

# Poll status
curl http://localhost:8001/job-status/<job_id>

# Download result
curl -O http://localhost:8001/download-video/<job_id>
```

---

## API Documentation

Base URL: `http://127.0.0.1:8001`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check — confirms models are loaded |
| POST | `/predict` | Predict Overall rating and market value |
| POST | `/injury-risk` | Assess injury risk category |
| POST | `/process-video` | Upload MP4, start async processing, return job ID (HTTP 202) |
| GET | `/job-status/{job_id}` | Poll job progress (0–100) and retrieve analytics on completion |
| GET | `/download-video/{job_id}` | Stream annotated MP4 (only available when status = "done") |

### Request / Response Schemas

**POST /predict — PlayerInput**
```json
{
  "age": 24,
  "potential": 85,
  "stamina": 80,
  "strength": 75,
  "sprint_speed": 90,
  "work_rate_encoded": 2
}
```
`work_rate_encoded`: integer 0–4, representing pandas categorical codes from the CSV `Work Rate` column.

**POST /injury-risk — InjuryInput**
```json
{"age": 28, "stamina": 70, "work_rate": 3}
```
`work_rate`: integer 1–5, used directly in the weighted risk formula (distinct from `work_rate_encoded`).

**GET /job-status response (completed job)**
```json
{
  "job_id": "...",
  "status": "done",
  "progress": 100,
  "analytics": {
    "players_detected": 14,
    "avg_speed_kmh": 8.3,
    "max_speed_kmh": 27.1,
    "total_distance_m": 4820.5,
    "frames_processed": 2700,
    "processing_time_s": 312.4
  }
}
```

---

## Project Structure

```
sports-analytics/
├── app/
│   ├── main.py       FastAPI app — lifespan model loading, job queue, all routes
│   ├── analyzer.py   SportsAnalyzer — preprocessing, model training, full video pipeline
│   └── schemas.py    Pydantic request schemas: PlayerInput, InjuryInput
├── data/
│   └── Footballer.csv   FIFA footballer dataset (not committed — 9.1 MB)
├── models/
│   ├── overall_model.pkl  XGBRegressor for Overall rating (auto-generated)
│   └── value_model.pkl    RandomForestRegressor for market value (auto-generated, ~250 MB)
├── output/           Runtime: staged uploads and processed output videos
├── frontend.py       Streamlit dashboard — 3 modules, async job polling
├── requirements.txt  Python dependencies (no pinned versions)
├── Dockerfile        Single-stage Python 3.10-slim + ffmpeg
├── AGENTS.md         Contributor and architecture reference
└── .gitignore        Custom ignore rules for this project
```

---

## Performance Considerations

- **Frame sampling**: YOLO runs on every 3rd frame (`frame_sample=3`). At 30 fps this cuts inference calls by 66%; skipped frames reuse the previous annotation, preserving smooth output video.
- **Inference resolution**: Each frame is resized internally to 640px width (`infer_width=640`). Higher values (e.g. 1280) improve detection of distant players at the cost of ~4x processing time.
- **Model variant**: YOLOv8 **nano** (`yolov8n.pt`) is the fastest and smallest variant. CPU-only inference on a typical laptop processes a 2-minute clip in 5–15 minutes.
- **Startup time**: `value_model.pkl` (~250 MB serialised) adds 3–5 seconds to backend startup even when models are pre-trained.
- **Concurrency**: The in-process job store is guarded by a single `threading.Lock`. Single-worker Uvicorn (`--workers 1`) is required; multi-worker deployments need an external job store.

---

## Known Limitations

- Only MP4 input is accepted at the upload endpoint.
- YOLOv8n misses heavily occluded or very small players (e.g. full-stadium wide-angle shots).
- Speed and distance estimation assumes a fixed broadcast pitch width of 105 metres. Angled or zoomed cameras produce inaccurate metric values.
- ByteTracker assigns a new ID after occlusions (ID switch). The `min_track_frames=20` filter removes short-lived stubs but may inflate player count in footage with frequent occlusions.
- Job state is in-process — lost on server restart. In-flight jobs become unrecoverable.
- No authentication on any API endpoint.
- `requirements.txt` has no pinned versions — future installs may differ.

---

## Future Improvements

- **Team separation** — classify players into two teams by jersey colour (K-means on HSV bounding box crops).
- **Player heatmaps** — kernel density estimation over centroid positions rendered on a pitch diagram.
- **Real-time streaming** — WebSocket + RTSP input using Ultralytics streaming mode.
- **Multi-camera re-identification** — appearance embeddings (BoT-SORT / OSNet) for cross-camera tracking.
- **Formation detection** — cluster team centroids per frame to infer tactical shape.
- **Pinned dependencies** — `pip freeze` for fully reproducible environments.
- **API authentication** — JWT tokens or API keys before any production deployment.

---

## Output Description

**Player Performance Prediction**: Two metric cards — Predicted Overall (0–99 scale matching FIFA ratings) and Market Value formatted in EUR.

**Injury Risk Assessment**: Colour-coded result banner (green = Low, amber = Medium, red = High) based on the formula: `risk = (age/40)×0.4 + (1−stamina/100)×0.4 + (work_rate/5)×0.2`.

**Video Tracking**: On completion, six metric cards (Players Detected, Avg Speed, Top Speed, Total Distance, Frames Processed, Processing Time) plus a download link for the annotated MP4. If `players_detected = 0`, an inline warning with diagnostic guidance appears directing the operator to check server logs for the `DIAGNOSTIC` log line.

---

## Conclusion

This platform demonstrates the integration of classical ML (gradient boosting, random forests) with modern computer vision (YOLOv8, ByteTracker) in a production-style architecture — async job queue, thread-safe state management, non-blocking UI. The video pipeline performs frame-level detection, multi-zone spatial filtering, persistent ID tracking, and kinematic metric computation entirely in Python without external annotation tooling.
