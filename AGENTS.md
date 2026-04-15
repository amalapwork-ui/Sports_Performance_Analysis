# Repository Guidelines

## Project Overview

Football player analytics platform combining ML-based performance prediction with YOLOv8 video player tracking. Split into two services that must run together:

- **FastAPI backend** (`app/`) — REST API on port 8001 (local) / 8000 (Docker)
- **Streamlit frontend** (`frontend.py`) — UI that calls the backend at `http://127.0.0.1:8001`

## Project Structure & Module Organization

```
app/
  main.py       # FastAPI app; loads models from models/ at startup
  analyzer.py   # SportsAnalyzer class: data preprocessing, model training, video processing
  schemas.py    # Pydantic request schemas: PlayerInput, InjuryInput
data/
  Footballer.csv  # Training data (FIFA footballer stats with currency columns: Value, Wage, Release Clause)
models/
  overall_model.pkl  # XGBRegressor — predicts player Overall rating
  value_model.pkl    # RandomForestRegressor — predicts market Value (EUR)
frontend.py     # Streamlit app; hardcoded to API_URL = "http://127.0.0.1:8001"
```

The API loads models from `models/` on startup. If `models/overall_model.pkl` does not exist, it trains and saves both models from `data/Footballer.csv` before serving requests. Models are **not** committed to git — they are generated on first run.

## Build & Development Commands

```bash
# Activate virtual environment (Windows)
source myvenv/Scripts/activate   # bash
myvenv\Scripts\activate.bat      # cmd

# Install dependencies
pip install -r requirements.txt

# Run backend (must start first)
uvicorn app.main:app --reload --port 8001

# Run frontend (in a separate terminal)
streamlit run frontend.py

# Docker (backend only, port 8000)
docker build -t sports-analytics .
docker run -p 8000:8000 sports-analytics
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predict Overall rating and market value |
| POST | `/injury-risk` | Assess injury risk (Low/Medium/High) |
| POST | `/process-video` | Upload MP4, run YOLOv8 tracking, save output to cwd |
| GET | `/job-status/{job_id}` | Poll async video processing job status |
| GET | `/download-video/{job_id}` | Stream processed video file by job ID |

## Coding Conventions

**Work rate encoding is inconsistent across endpoints — this is intentional:**
- `PlayerInput.work_rate_encoded`: integer 0–4 (pandas categorical codes from the CSV)
- `InjuryInput.work_rate`: integer 1–5 (used directly in the risk arithmetic formula)

Do not unify these without updating both `analyzer.py` and `frontend.py`.

Currency columns in the CSV (`Value`, `Wage`, `Release Clause`) use strings like `€110M` or `€50K`. `SportsAnalyzer._clean_currency()` converts them via `eval()` — do not change the data format without updating this method.

Models are trained using these features (column names must match CSV exactly):
```python
["Age", "Potential", "Stamina", "Strength", "SprintSpeed", "Work Rate Encoded"]
```

## Dependencies

Python 3.10 (see `Dockerfile`). Key packages: `fastapi`, `uvicorn`, `streamlit`, `scikit-learn`, `xgboost`, `joblib`, `ultralytics` (YOLOv8), `opencv-python`, `pandas`, `numpy`, `python-multipart`.

No test framework, linter, or formatter is configured.
