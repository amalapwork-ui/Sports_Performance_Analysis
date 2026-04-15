# Operation Manual — Sports Performance Analytics Platform

This manual covers system startup, module usage, input requirements, expected outputs, error handling, and troubleshooting. Intended for operators and end-users who run the system locally.

---

## System Overview

The platform consists of two independent processes that must both be running to use any feature:

| Process | Command | Port | Purpose |
|---|---|---|---|
| Backend (FastAPI) | `uvicorn app.main:app --reload --port 8001` | 8001 | Serves ML predictions and processes video |
| Frontend (Streamlit) | `streamlit run frontend.py` | 8501 | Browser dashboard for interacting with the system |

The frontend calls the backend over HTTP. Neither process is useful without the other. Always start the backend first.

---

## How to Run the System

### Step 1 — Activate the virtual environment

```bash
# Windows (Git Bash / bash)
source myvenv/Scripts/activate

# Windows (cmd.exe)
myvenv\Scripts\activate.bat

# macOS / Linux
source myvenv/bin/activate
```

You should see `(myvenv)` at the start of your terminal prompt.

### Step 2 — Start the backend

Open a terminal and run:

```bash
uvicorn app.main:app --reload --port 8001
```

Watch the console output. On first run (no model files in `models/`), you will see:

```
INFO - No saved models — training now (may take a minute)...
INFO - Training complete.
INFO - Models loaded. API is ready.
```

The API is not ready until you see `Models loaded. API is ready.` This takes 60–120 seconds on the first run; subsequent starts take 5–10 seconds.

### Step 3 — Start the frontend

Open a second terminal (leave the backend running), activate the virtual environment again, then run:

```bash
streamlit run frontend.py
```

Streamlit will print a local URL, typically `http://localhost:8501`. Open it in your browser.

### Step 4 — Verify connection

The top of the dashboard will show a yellow warning banner if the backend is offline. If the banner is absent, the system is ready.

---

## How to Use Each Module

Use the **sidebar** on the left to switch between modules.

---

### Module 1 — Player Performance Prediction

**Purpose**: Predict a player's FIFA Overall rating (0–99) and estimated market value (EUR) from physical attributes.

**Inputs** (all set via sliders and dropdown):

| Field | Range | Description |
|---|---|---|
| Age | 16–40 | Player's age in years |
| Potential | 50–95 | FIFA potential rating |
| Stamina | 30–99 | Physical endurance rating |
| Strength | 30–99 | Physical strength rating |
| Sprint Speed | 30–99 | Sprinting ability rating |
| Work Rate (Encoded) | 0, 1, 2, 3, 4 | Categorical code for work rate (from FIFA dataset) |

Work Rate Encoded values correspond to FIFA dataset categories. If you are unsure, use 2 as a neutral mid-range value.

**How to use**:
1. Adjust all sliders to match the player's known attributes.
2. Click **Predict Performance**.
3. Results appear as two metric cards below the button.

**Expected output**:
```
Predicted Overall:   82.45
Market Value (€):    42,300,000
```

---

### Module 2 — Injury Risk Assessment

**Purpose**: Classify a player's injury risk as Low, Medium, or High based on three inputs.

**Inputs**:

| Field | Range | Description |
|---|---|---|
| Age | 16–40 | Player's age |
| Stamina | 30–99 | Physical endurance rating |
| Work Rate | 1–5 | Activity level (1 = lowest, 5 = highest) |

Note: Work Rate here is a 1–5 integer, not the encoded 0–4 value used in the Prediction module. These are intentionally different scales.

**Risk formula** (for reference):
```
risk_score = (age / 40) × 0.4 + (1 − stamina / 100) × 0.4 + (work_rate / 5) × 0.2

Low    →  risk_score < 0.33
Medium →  0.33 ≤ risk_score < 0.66
High   →  risk_score ≥ 0.66
```

**How to use**:
1. Set Age, Stamina, and Work Rate.
2. Click **Assess Injury Risk**.
3. Result appears as a colour-coded banner:
   - Green panel: Low Risk
   - Yellow/amber panel: Medium Risk
   - Red panel: High Risk

---

### Module 3 — Video Player Tracking

**Purpose**: Upload a football match video, automatically detect and track all players using YOLOv8, and receive movement analytics (speed, distance) plus an annotated output video.

**How to use**:

1. Click **Browse files** or drag and drop an MP4 file onto the uploader.
2. Click **Process Video**. The video uploads immediately; the page shows "Uploading video..." briefly.
3. Processing starts in the background. A progress bar and status message appear:
   - 0–20%: "Detecting and tracking players..."
   - 20–50%: "Analyzing player movements..."
   - 50–80%: "Computing speed and distance metrics..."
   - 80–100%: "Finalizing annotated video..."
4. When complete, the dashboard shows:
   - A download link for the annotated MP4
   - Six analytics metric cards
5. Click **Process Another Video** to reset and start again.

**Processing time**: Approximately 2–5 minutes per minute of video on CPU hardware. Do not close the browser tab while processing — the Streamlit polling loop must stay active.

---

## Input Requirements

### Video files

| Requirement | Value |
|---|---|
| Format | MP4 only (.mp4 extension) |
| Recommended resolution | 720p or 1080p broadcast footage |
| Camera angle | Wide broadcast angle showing full pitch width |
| Recommended clip length | 30 seconds to 5 minutes |
| Maximum clip length | No hard limit; processing time scales linearly |

**Inputs that produce poor results**:
- Close-up camera shots (players too large, pitch scale incorrect)
- Handheld or phone camera footage (excessive motion blur)
- Night matches with poor lighting
- Clips shorter than ~10 seconds (tracker needs consecutive frames to initialise)

---

## Expected Outputs

### Video tracking analytics

| Metric | Unit | Meaning |
|---|---|---|
| Players Detected | count | Unique valid track IDs surviving the 20-frame minimum filter |
| Avg Speed | km/h | Mean of per-player average speeds |
| Top Speed | km/h | Highest instantaneous speed recorded across all players |
| Total Distance | metres | Sum of distances covered by all tracked players |
| Frames Processed | count | Total frames written to output video |
| Processing Time | seconds | Wall-clock time for the full video pipeline |

Speeds are clamped to the physiologically realistic range of 0.5–40 km/h. Readings outside this range are discarded as tracking noise.

### Annotated video

The output MP4 contains the original footage with:
- Coloured bounding boxes around each detected person
- A numeric track ID above each box
- Frame-rate and resolution matching the source video

---

## Error Handling

### "Backend is offline or still loading models"

Shown as a yellow banner at the top of the dashboard.

**Cause**: The backend is either not running, still training models on first startup, or starting up.

**Action**: Check the backend terminal. Wait for `Models loaded. API is ready.` before using the frontend.

---

### "Cannot reach the backend"

Shown as a red error box after clicking any action button.

**Cause**: The backend process crashed, was stopped, or is not running on port 8001.

**Action**: Go to Terminal 1 and restart: `uvicorn app.main:app --reload --port 8001`

---

### "Only MP4 videos are supported"

**Cause**: A file with a non-MP4 extension was uploaded.

**Action**: Convert the video to MP4 format before uploading (e.g. using VLC or HandBrake).

---

### "Request timed out"

Shown after the upload step.

**Cause**: The video file is very large and the upload took longer than 60 seconds.

**Action**: Use a shorter or lower-resolution clip. The 60-second upload timeout is set in the frontend `call_api` helper.

---

### "Processing failed: ..."

Shown after the progress bar, with the error message from the backend.

**Cause**: An unexpected error in the video pipeline (corrupted file, unsupported codec, out of memory).

**Action**: Check the backend terminal for the full traceback. Try a different video file.

---

### "No players were tracked in this video"

Shown on the analytics dashboard when `players_detected = 0`.

**Cause**: YOLO found no detections, or all detections were filtered out. Common causes:
- Players are too small in frame (stadium overview shot)
- Poor lighting or heavy motion blur
- Non-broadcast camera angle where players do not appear as recognisable persons
- Video clip too short for the ByteTracker to initialise (fewer than ~60 frames)

**Action**: Check the backend console for a line containing `DIAGNOSTIC`. The log specifies whether the issue is zero detections (YOLO not seeing anything) or zero tracked frames (detections exist but ByteTracker failed to initialise).

---

## Troubleshooting Guide

### Backend will not start

1. Confirm the virtual environment is activated (`(myvenv)` in prompt).
2. Run `pip install -r requirements.txt` to ensure all packages are present.
3. Confirm `data/Footballer.csv` exists — the backend reads it on startup.
4. Check port 8001 is not already in use: on Windows run `netstat -ano | findstr 8001`.

---

### Models are missing after training completes

The backend trains models into `models/overall_model.pkl` and `models/value_model.pkl`. If `models/` does not exist, the write will fail silently.

**Fix**:
```bash
mkdir models
uvicorn app.main:app --reload --port 8001
```

---

### Frontend shows connection error immediately after backend starts

The backend needs 60–120 seconds to train models on first run. The frontend's health check (`/`) returns `"models_loaded": false` during training, which triggers the warning banner. Wait for the `Models loaded. API is ready.` log line before interacting.

---

### Video processing produces all-zero metrics

All six analytics values are zero. This means tracking ran but no valid player tracks were retained after the 20-frame minimum filter.

Check the backend log for:
```
DIAGNOSTIC — YOLO found 0 detections across N inferred frames.
```
or:
```
DIAGNOSTIC — YOLO detected N objects ... but ByteTracker assigned 0 track IDs.
```

The first message means YOLO is not detecting anything — try a clearer, wider broadcast-angle clip. The second message means detections exist but the clip is too short for ByteTracker to build stable tracks — use a longer clip.

---

### Processing is very slow

YOLOv8n runs on CPU by default. For a 2-minute clip at 1080p, expect 5–15 minutes on a mid-range laptop. This is expected behaviour.

**Options to speed up**:
- Use a shorter clip (30–60 seconds)
- Use 720p footage instead of 1080p
- If a CUDA-capable GPU is available, install `torch` with CUDA support before `ultralytics`. YOLO will automatically use the GPU.

---

## Best Practices

- **Use broadcast match footage** — wide-angle shots where all players are consistently visible. Zoomed or close-up shots distort the 105-metre pitch-width calibration and produce incorrect speed readings.
- **Keep clips under 3 minutes** for initial testing — this gives a good result set without long waits.
- **Do not refresh the browser** while processing — the Streamlit session state holds the job ID, and refreshing loses it. The backend job continues running; you can retrieve results via the API directly if needed using `curl http://localhost:8001/job-status/<job_id>`.
- **Restart the backend** between very different videos if you observe inflated player counts — the tracker state is reset per job, but any crash mid-job may leave the YOLO predictor in a bad state.
- **Train once, reuse** — after the first successful start, `models/*.pkl` files persist. Subsequent restarts load them in ~5 seconds.
