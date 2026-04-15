import time

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8001"
POLL_INTERVAL = 3  # seconds between job-status polls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_backend() -> bool:
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.status_code == 200 and r.json().get("models_loaded", False)
    except requests.exceptions.RequestException:
        return False


def call_api(method: str, path: str, timeout: int = 30, **kwargs):
    """
    Thin requests wrapper. Returns Response on success (200/202),
    None on any failure (shows Streamlit error messages inline).
    """
    try:
        response = getattr(requests, method)(
            f"{API_URL}{path}", timeout=timeout, **kwargs
        )
        if response.status_code not in (200, 202):
            st.error(f"API error {response.status_code}: {response.text}")
            return None
        return response
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the backend. "
            "Make sure the server is running:\n\n"
            "`uvicorn app.main:app --port 8001`"
        )
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
        return None


def _progress_message(pct: int) -> str:
    """Map processing percentage to a human-readable status string."""
    if pct < 20:
        return "Detecting and tracking players..."
    if pct < 50:
        return "Analyzing player movements..."
    if pct < 80:
        return "Computing speed and distance metrics..."
    return "Finalizing annotated video..."


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Sports Analytics Platform", layout="wide")
st.title("Sports Performance Analytics")
st.markdown("ML-powered football player analysis dashboard")

if not check_backend():
    st.warning(
        "Backend is offline or still loading models. "
        "Start it with: `uvicorn app.main:app --port 8001`"
    )

menu = st.sidebar.radio(
    "Select Module",
    ["Player Performance Prediction", "Injury Risk Assessment", "Video Player Tracking"],
)

# ---------------------------------------------------------------------------
# Player Performance Prediction
# ---------------------------------------------------------------------------
if menu == "Player Performance Prediction":
    st.header("Player Performance Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 16, 40, 24)
        potential = st.slider("Potential", 50, 95, 85)
        stamina = st.slider("Stamina", 30, 99, 80)
    with col2:
        strength = st.slider("Strength", 30, 99, 75)
        sprint_speed = st.slider("Sprint Speed", 30, 99, 90)
        work_rate_encoded = st.selectbox("Work Rate (Encoded)", [0, 1, 2, 3, 4])

    if st.button("Predict Performance"):
        payload = {
            "age": age,
            "potential": potential,
            "stamina": stamina,
            "strength": strength,
            "sprint_speed": sprint_speed,
            "work_rate_encoded": work_rate_encoded,
        }
        response = call_api("post", "/predict", json=payload)
        if response:
            result = response.json()
            st.success("Prediction successful")
            st.metric("Predicted Overall", result["predicted_overall"])
            st.metric("Market Value (€)", f"{result['predicted_value_eur']:,.0f}")

# ---------------------------------------------------------------------------
# Injury Risk Assessment
# ---------------------------------------------------------------------------
elif menu == "Injury Risk Assessment":
    st.header("Injury Risk Assessment")

    age = st.slider("Age", 16, 40, 28)
    stamina = st.slider("Stamina", 30, 99, 70)
    work_rate = st.selectbox("Work Rate", [1, 2, 3, 4, 5])

    if st.button("Assess Injury Risk"):
        payload = {"age": age, "stamina": stamina, "work_rate": work_rate}
        response = call_api("post", "/injury-risk", json=payload)
        if response:
            risk = response.json()["injury_risk"]
            if risk == "High":
                st.error("High Injury Risk")
            elif risk == "Medium":
                st.warning("Medium Injury Risk")
            else:
                st.success("Low Injury Risk")

# ---------------------------------------------------------------------------
# Video Player Tracking  — job-based, non-blocking
# ---------------------------------------------------------------------------
elif menu == "Video Player Tracking":
    st.header("Player Detection & Tracking (YOLOv8)")

    # Session state initialisation
    for key, default in [
        ("video_job_id", None),
        ("video_status", None),
        ("video_analytics", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    uploaded_file = st.file_uploader("Upload Football Match Video", type=["mp4"])

    # ── (A) No active job → show submit button ──────────────────────────────
    if uploaded_file and st.session_state.video_job_id is None:
        if st.button("Process Video"):
            with st.spinner("Uploading video..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "video/mp4",
                    )
                }
                response = call_api(
                    "post", "/process-video", timeout=60, files=files
                )

            if response:
                st.session_state.video_job_id = response.json()["job_id"]
                st.session_state.video_status = "pending"
                st.session_state.video_analytics = None
                st.rerun()

    # ── (B) Job running → poll, show human-readable progress ────────────────
    if st.session_state.video_job_id and st.session_state.video_status not in (
        "done", "failed", None
    ):
        progress_bar = st.progress(0)
        status_text = st.empty()

        while True:
            resp = call_api(
                "get",
                f"/job-status/{st.session_state.video_job_id}",
                timeout=5,
            )
            if resp is None:
                st.session_state.video_job_id = None
                st.session_state.video_status = None
                break

            data = resp.json()
            pct = data.get("progress", 0)
            status = data["status"]
            st.session_state.video_status = status

            progress_bar.progress(pct / 100)
            status_text.text(_progress_message(pct))

            if status == "done":
                # Break first, then make one dedicated fetch for analytics.
                # This guarantees we read a fully-committed job snapshot rather
                # than whichever poll response happened to carry status="done".
                break

            if status == "failed":
                st.error(
                    f"Processing failed: {data.get('error', 'Unknown error')}"
                )
                st.session_state.video_job_id = None
                st.session_state.video_status = None
                break

            time.sleep(POLL_INTERVAL)

    # After polling ends with "done", fetch the final job snapshot once more
    # to guarantee analytics is present (polling response may have been stale).
    if (
        st.session_state.video_status == "done"
        and st.session_state.video_job_id
        and st.session_state.video_analytics is None
    ):
        final = call_api(
            "get",
            f"/job-status/{st.session_state.video_job_id}",
            timeout=10,
        )
        if final:
            st.session_state.video_analytics = final.json().get("analytics")

    # ── (C) Job complete → analytics dashboard ──────────────────────────────
    if (
        st.session_state.video_status == "done"
        and st.session_state.video_job_id
    ):
        st.success("Analysis complete!")
        st.divider()

        download_url = (
            f"{API_URL}/download-video/{st.session_state.video_job_id}"
        )
        col_video, col_stats = st.columns([3, 2], gap="large")

        with col_video:
            st.subheader("Processed Video")
            st.markdown(
                f"**[Download annotated video]({download_url})**",
                unsafe_allow_html=False,
            )
            st.caption(
                "Open the link in your browser to play or save the video."
            )

        with col_stats:
            st.subheader("Player Analytics")
            a = st.session_state.video_analytics  # None = not loaded; {} = no data

            if a is None:
                # Should not reach here after the explicit fetch above,
                # but guard anyway so the UI always shows something.
                st.warning(
                    "Analytics data could not be loaded. "
                    "The video was processed — check the server logs for details."
                )
            else:
                # Always render the six metrics, even when values are zero.
                # Zero values are meaningful: they indicate tracking did not fire.
                row1_left, row1_right = st.columns(2)
                row2_left, row2_right = st.columns(2)
                row3_left, row3_right = st.columns(2)

                with row1_left:
                    st.metric("Players Detected", a.get("players_detected", 0))
                with row1_right:
                    avg = a.get("avg_speed_kmh", 0)
                    st.metric("Avg Speed", f"{avg} km/h")

                with row2_left:
                    top = a.get("max_speed_kmh", 0)
                    st.metric("Top Speed", f"{top} km/h")
                with row2_right:
                    dist = a.get("total_distance_m", 0)
                    st.metric("Total Distance", f"{dist:,.0f} m")

                with row3_left:
                    frames = a.get("frames_processed", 0)
                    st.metric("Frames Processed", f"{frames:,}")
                with row3_right:
                    secs = a.get("processing_time_s", 0)
                    st.metric("Processing Time", f"{secs} s")

                # Contextual guidance when tracking produced no results
                if a.get("players_detected", 0) == 0:
                    st.warning(
                        "No players were tracked in this video. "
                        "Check the server logs for a DIAGNOSTIC line. "
                        "Common causes: players too small in frame, "
                        "poor lighting, or a non-broadcast camera angle."
                    )

        st.divider()
        if st.button("Process Another Video"):
            st.session_state.video_job_id = None
            st.session_state.video_status = None
            st.session_state.video_analytics = None
            st.rerun()
