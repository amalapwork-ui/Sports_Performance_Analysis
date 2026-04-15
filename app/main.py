from contextlib import asynccontextmanager
import logging
import os
import shutil
import threading
import uuid

import joblib
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

from app.analyzer import SportsAnalyzer
from app.schemas import PlayerInput, InjuryInput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

analyzer: SportsAnalyzer | None = None

# ---------------------------------------------------------------------------
# In-process job store
# Each entry: {status, progress, output_path, error, filename, analytics}
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

OUTPUT_DIR = "output"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize heavy resources after the port is bound."""
    global analyzer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Loading SportsAnalyzer and ML models...")
    analyzer = SportsAnalyzer("data/Footballer.csv")

    if not os.path.exists("models/overall_model.pkl"):
        logger.info("No saved models — training now (may take a minute)...")
        analyzer.train_and_save_models()
        logger.info("Training complete.")

    analyzer.overall_model = joblib.load("models/overall_model.pkl")
    analyzer.value_model = joblib.load("models/value_model.pkl")
    logger.info("Models loaded. API is ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Sports Analytics API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Background video worker
# ---------------------------------------------------------------------------

def _run_video_job(job_id: str, staging_path: str, filename: str) -> None:
    """
    Runs in a FastAPI BackgroundTasks thread.
    Processes the staged file, collects analytics, updates job state, cleans up.
    """
    output_path = os.path.join(OUTPUT_DIR, f"processed_{filename}")

    def on_progress(pct: int) -> None:
        with jobs_lock:
            jobs[job_id]["progress"] = pct

    try:
        with jobs_lock:
            jobs[job_id]["status"] = "processing"

        analytics = analyzer.process_video(
            staging_path, output_path, progress_cb=on_progress
        )

        with jobs_lock:
            jobs[job_id].update(
                status="done",
                progress=100,
                output_path=output_path,
                analytics=analytics,
            )
        logger.info(
            "Job %s complete — players=%d, avg_speed=%.1f km/h → %s",
            job_id,
            analytics.get("players_detected", 0),
            analytics.get("avg_speed_kmh", 0.0),
            output_path,
        )

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        with jobs_lock:
            jobs[job_id].update(status="failed", error=str(exc))

    finally:
        if os.path.exists(staging_path):
            os.remove(staging_path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "ok", "models_loaded": analyzer is not None}


@app.post("/predict")
def predict_player(data: PlayerInput):
    features = [
        data.age, data.potential, data.stamina,
        data.strength, data.sprint_speed,
        data.work_rate_encoded,
    ]
    overall, value = analyzer.predict(features)
    return {
        "predicted_overall": float(round(overall, 2)),
        "predicted_value_eur": float(round(value, 2)),
    }


@app.post("/injury-risk")
def injury_risk(data: InjuryInput):
    return {
        "injury_risk": analyzer.assess_injury_risk(
            data.age, data.stamina, data.work_rate
        )
    }


@app.post("/process-video", status_code=202)
def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accepts the upload, saves to staging, returns job_id immediately (202).
    Processing runs in a background thread — poll /job-status/{job_id}.
    """
    if not file.filename.lower().endswith(".mp4"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only MP4 videos are supported"},
        )

    job_id = str(uuid.uuid4())

    # Persist the upload before returning — UploadFile is closed after response.
    staging_path = os.path.join(OUTPUT_DIR, f"staging_{job_id}_{file.filename}")
    with open(staging_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "output_path": None,
            "error": None,
            "filename": file.filename,
            "analytics": None,
        }

    background_tasks.add_task(_run_video_job, job_id, staging_path, file.filename)
    logger.info("Job %s queued for %s", job_id, file.filename)
    return {"job_id": job_id, "status": "pending"}


@app.get("/job-status/{job_id}")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        # Copy inside the lock so the unpack below reads a consistent snapshot,
        # never a partially-written dict if the background task is mid-update.
        job_snapshot = dict(job)
    return {"job_id": job_id, **job_snapshot}


@app.get("/download-video/{job_id}")
def download_video(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Video not ready yet")
    path = job["output_path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Output file missing on disk")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=os.path.basename(path),
    )
