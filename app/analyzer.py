import logging
import time

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ultralytics import YOLO
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

# Broadcast football matches typically show the full pitch width (~105 m).
# This constant converts pixel displacements to metres for speed estimation.
_PITCH_WIDTH_M = 105.0

# Physiologically realistic speed range for a footballer on camera.
# Observations outside this band are discarded as tracking noise.
_MIN_SPEED_KMH = 0.5
_MAX_SPEED_KMH = 40.0


class SportsAnalyzer:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.yolo = YOLO("yolov8n.pt")
        self._preprocess()

    # ------------------------------------------------------------------
    # ML model helpers
    # ------------------------------------------------------------------

    def _clean_currency(self, val):
        if isinstance(val, str):
            val = val.replace("€", "").replace("M", "e6").replace("K", "e3")
            return eval(val)
        return val

    def _preprocess(self):
        for col in ["Value", "Wage", "Release Clause"]:
            self.df[col] = self.df[col].apply(self._clean_currency)
        self.df.fillna(self.df.median(numeric_only=True), inplace=True)
        self.df["Work Rate Encoded"] = (
            self.df["Work Rate"].astype("category").cat.codes
        )

    def train_and_save_models(self):
        features = [
            "Age", "Potential", "Stamina",
            "Strength", "SprintSpeed", "Work Rate Encoded",
        ]
        X = self.df[features]
        y_overall = self.df["Overall"]
        y_value = self.df["Value"]

        overall_model = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6
        )
        overall_model.fit(X, y_overall)

        value_model = RandomForestRegressor(n_estimators=200)
        value_model.fit(X, y_value)

        joblib.dump(overall_model, "models/overall_model.pkl")
        joblib.dump(value_model, "models/value_model.pkl")

    def predict(self, features):
        overall = float(self.overall_model.predict([features])[0])
        value = float(self.value_model.predict([features])[0])
        return overall, value

    def assess_injury_risk(self, age, stamina, work_rate):
        risk_score = (
            (age / 40) * 0.4
            + (1 - stamina / 100) * 0.4
            + (work_rate / 5) * 0.2
        )
        if risk_score < 0.33:
            return "Low"
        elif risk_score < 0.66:
            return "Medium"
        return "High"

    # ------------------------------------------------------------------
    # Video processing + analytics
    # ------------------------------------------------------------------

    def process_video(
        self,
        input_path: str,
        output_path: str,
        frame_sample: int = 3,
        infer_width: int = 640,
        min_track_frames: int = 20,
        conf_threshold: float = 0.60,
        progress_cb=None,
    ) -> dict:
        """
        Run YOLOv8 tracking on a video, write the annotated output, and
        return a dict of aggregated player analytics.

        Parameters
        ----------
        frame_sample      Run YOLO on every Nth frame; copy last annotation
                          for skipped frames (3× throughput at 30 fps).
        infer_width       YOLO internal resize width. Boxes are scaled back to
                          source resolution before plot().
        min_track_frames  A track ID must appear in at least this many inferred
                          frames to be counted as a real player.  Eliminates:
                            - Spurious 1–few-frame detections (scoreboards,
                              crowd glimpses, advertisement boards).
                            - ID-switch stubs: when a player is occluded and
                              ByteTracker assigns a new ID on re-detection,
                              both the old tail and the new head are short.
                          Default 20 ≈ 2 seconds of continuous tracking at
                          30 fps / frame_sample=3.
        conf_threshold    Minimum YOLO detection confidence (0–1).  The YOLO
                          default of 0.25 accepts many crowd/staff false
                          positives; 0.50 keeps only clearly-visible persons.
        progress_cb       Optional callable(int 0–100) fired at 10% intervals.
        """
        # ── Reset tracker state from any previous video job ───────────────
        # self.yolo is a shared singleton.  ByteTracker stores its state in
        # self.yolo.predictor.trackers[0].  If we don't reset it, IDs from
        # the last video remain "active" and contaminate the new video's
        # track assignments, inflating the player count by 20-40 ghost IDs.
        self.yolo.predictor = None

        t_start = time.time()

        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (vid_w, vid_h),
        )

        # pixels → metres conversion (broadcast pitch-width assumption)
        px_per_m = vid_w / _PITCH_WIDTH_M if vid_w > 0 else 1.0

        # track_history: {track_id: [(frame_idx, cx, cy), ...]}
        # Only inferred frames contribute entries — skipped frames are silent.
        track_history: dict[int, list[tuple[int, float, float]]] = {}

        log_interval = max(1, total_frames // 10) if total_frames else 100
        frame_idx = 0
        last_annotated = None

        # Diagnostic counters
        _inferred_frames = 0    # frames where yolo.track() ran
        _total_detections = 0   # raw person-class boxes across all inferred frames
        _tracked_frames = 0     # inferred frames where tracker assigned IDs

        # Vertical crop bounds: discard detections above (crowd stands) or below
        # (pitch-side ad boards) the playing area.  Expressed as fractions of
        # frame height.  Top 15% is typically stands; bottom 8% ad boards.
        _crop_top = int(vid_h * 0.15)
        _crop_bot = int(vid_h * 0.92)

        # Box-height bounds: in a broadcast pitch-wide shot, a player's box is
        # typically 5–28% of frame height.  Smaller → crowd member or noise;
        # larger → close-up of coach / substitute on the bench.
        _min_box_h = vid_h * 0.05
        _max_box_h = vid_h * 0.28

        logger.info(
            "process_video start — %d frames @ %d fps, "
            "sample every %d, infer_width=%d, min_track_frames=%d, conf=%.2f",
            total_frames, fps, frame_sample, infer_width, min_track_frames,
            conf_threshold,
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_sample == 0:
                results = self.yolo.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=conf_threshold,
                    imgsz=infer_width,
                    verbose=False,
                )
                last_annotated = results[0].plot()
                _inferred_frames += 1

                boxes = results[0].boxes
                if boxes is not None:
                    _total_detections += len(boxes)
                    if boxes.id is not None:
                        _tracked_frames += 1
                        coords = boxes.xyxy.cpu().numpy()
                        ids = boxes.id.cpu().numpy().astype(int)
                        for (x1, y1, x2, y2), tid in zip(coords, ids):
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            box_h = y2 - y1
                            # Skip detections in crowd/ad-board zones
                            if cy < _crop_top or cy > _crop_bot:
                                continue
                            # Skip boxes that are too small (crowd noise) or
                            # too large (close-up coach / substitute)
                            if box_h < _min_box_h or box_h > _max_box_h:
                                continue
                            track_history.setdefault(int(tid), []).append(
                                (frame_idx, cx, cy)
                            )

            out.write(last_annotated if last_annotated is not None else frame)
            frame_idx += 1

            if frame_idx % log_interval == 0:
                pct = int(frame_idx / total_frames * 100) if total_frames else 0
                logger.info(
                    "progress %d%% — %d/%d frames | "
                    "inferred=%d, detections=%d, tracked_frames=%d, raw_ids=%d",
                    pct, frame_idx, total_frames,
                    _inferred_frames, _total_detections,
                    _tracked_frames, len(track_history),
                )
                if progress_cb:
                    progress_cb(pct)

        cap.release()
        out.release()

        processing_time = round(time.time() - t_start, 1)
        logger.info(
            "process_video complete in %.1f s → %s", processing_time, output_path
        )

        # ── Step 1: Filter short-lived tracks ─────────────────────────────
        # Remove any track whose lifetime (number of inferred frames it
        # appeared in) is below min_track_frames.  This eliminates:
        #   a) Spurious detections: scoreboards, crowd glimpses, jersey
        #      numbers momentarily classified as persons.
        #   b) ID-switch stubs: ByteTracker creates a new ID when a player
        #      is briefly occluded.  The stub for the old ID (last few frames)
        #      and the stub for the new ID (first few frames) are both short.
        raw_track_count = len(track_history)
        valid_tracks = {
            tid: positions
            for tid, positions in track_history.items()
            if len(positions) >= min_track_frames
        }
        removed = raw_track_count - len(valid_tracks)

        logger.info(
            "Track filter (conf=%.2f, min_frames=%d): %d raw IDs → %d valid, removed %d stubs",
            conf_threshold, min_track_frames, raw_track_count, len(valid_tracks), removed,
        )

        # ── Step 2: Per-player analytics ──────────────────────────────────
        # Compute average speed, top speed, and total distance per player,
        # then aggregate across players.
        #
        # avg_speed = mean of per-player averages (not global mean of all
        # instantaneous observations — that over-weights players who appear
        # in more frames).
        player_avg_speeds: list[float] = []   # one per valid track
        player_top_speeds: list[float] = []   # one per valid track
        player_distances: list[float] = []    # total distance per track (m)

        for tid, positions in valid_tracks.items():
            if len(positions) < 2:
                continue

            total_dist_px = 0.0
            step_speeds: list[float] = []

            for i in range(1, len(positions)):
                f0, x0, y0 = positions[i - 1]
                f1, x1, y1 = positions[i]
                dt = (f1 - f0) / fps      # seconds between these two detections
                if dt <= 0:
                    continue
                dist_px = float(np.hypot(x1 - x0, y1 - y0))
                total_dist_px += dist_px
                speed_kmh = (dist_px / px_per_m / dt) * 3.6
                # Discard physiologically impossible values
                if _MIN_SPEED_KMH <= speed_kmh <= _MAX_SPEED_KMH:
                    step_speeds.append(speed_kmh)

            if step_speeds:
                player_avg_speeds.append(float(np.mean(step_speeds)))
                player_top_speeds.append(float(np.max(step_speeds)))

            player_distances.append(total_dist_px / px_per_m)

        avg_speed = (
            round(float(np.mean(player_avg_speeds)), 1)
            if player_avg_speeds else 0.0
        )
        max_speed = (
            round(float(np.max(player_top_speeds)), 1)
            if player_top_speeds else 0.0
        )
        total_distance = (
            round(float(np.sum(player_distances)), 1)
            if player_distances else 0.0
        )

        # ── Diagnostic summary ─────────────────────────────────────────────
        if _tracked_frames == 0 and _inferred_frames > 0:
            if _total_detections == 0:
                logger.warning(
                    "DIAGNOSTIC — YOLO found 0 detections across %d inferred frames. "
                    "Possible causes: wrong class filter, video too dark/blurry, "
                    "players too small for yolov8n at imgsz=%d. "
                    "Try a clearer video or increase infer_width to 1280.",
                    _inferred_frames, infer_width,
                )
            else:
                logger.warning(
                    "DIAGNOSTIC — YOLO detected %d objects across %d inferred frames "
                    "but ByteTracker assigned 0 track IDs. "
                    "Tracker needs more consecutive frames to initialise. "
                    "All speed/distance metrics will be 0.",
                    _total_detections, _inferred_frames,
                )
        else:
            logger.info(
                "DIAGNOSTIC — inferred=%d, detections=%d, tracked_frames=%d (%.0f%%), "
                "raw_ids=%d, valid_ids=%d (filtered %d stubs)",
                _inferred_frames, _total_detections, _tracked_frames,
                100 * _tracked_frames / _inferred_frames if _inferred_frames else 0,
                raw_track_count, len(valid_tracks), removed,
            )

        result = {
            "players_detected": len(valid_tracks),
            "avg_speed_kmh": avg_speed,
            "max_speed_kmh": max_speed,
            "total_distance_m": total_distance,
            "frames_processed": frame_idx,
            "processing_time_s": processing_time,
        }
        logger.info(
            "Analytics result — players=%d, avg=%.1f km/h, top=%.1f km/h, "
            "dist=%.1f m, frames=%d, time=%.1f s",
            result["players_detected"],
            result["avg_speed_kmh"],
            result["max_speed_kmh"],
            result["total_distance_m"],
            result["frames_processed"],
            result["processing_time_s"],
        )

        # Reset tracker so the next job starts with a clean slate
        self.yolo.predictor = None

        return result
