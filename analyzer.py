"""
Core analysis pipeline — shared by the command line (main.py) and the web
app (app.py), so there's only ever one copy of this logic.

The important difference from a plain script: run_analysis() reports progress
through a callback and writes everything for a single run into its own folder
under runs/, so past matches stay browsable instead of being overwritten.
"""
import os
import json
import shutil
import subprocess
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from team_assigner import TeamAssigner

POSSESSION_MAX_DIST = 80  # pixels; ball must be within this of a player to count as "possessed"
FALLBACK_TEAM_BGR = {1: (0, 0, 255), 2: (255, 100, 0)}  # only used before teams are worked out
TIMELINE_BUCKETS = 48     # how many segments the possession-over-time strip is split into

RUNS_DIR = "runs"


def detect_relevant_classes(model, log=print):
    """Works out which class IDs mean 'a player' and which mean 'the ball',
    by name, so this works with BOTH the generic pretrained COCO model
    (class 'person', 'sports ball') AND a custom-trained football model
    (classes 'player', 'goalkeeper', 'ball', 'referee' — referees are
    deliberately excluded, same as we do for non-bib people)."""
    names = model.names  # {id: name}
    person_ids, ball_ids = [], []
    for class_id, name in names.items():
        lname = name.lower()
        if "referee" in lname:
            continue
        if "ball" in lname:
            ball_ids.append(class_id)
        elif "person" in lname or "player" in lname or "goalkeeper" in lname:
            person_ids.append(class_id)

    if not person_ids or not ball_ids:
        raise RuntimeError(
            f"Couldn't find player/ball classes in this model's class list: {names}. "
            "Expected something like 'person'/'ball' or 'player'/'ball'."
        )

    log(f"Detected classes — players: {[names[i] for i in person_ids]}, "
        f"ball: {[names[i] for i in ball_ids]}")
    return person_ids, ball_ids


def bbox_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def load_roi(roi_path="roi.json", log=print):
    if roi_path and os.path.exists(roi_path):
        with open(roi_path) as f:
            roi = json.load(f)
        if "polygon" in roi:
            log(f"Using polygon pitch boundary ({len(roi['polygon'])} points)")
            return {"type": "polygon", "points": np.array(roi["polygon"], dtype=np.int32)}
        log("Using rectangle crop area")
        return {"type": "rect", "x1": roi["x1"], "y1": roi["y1"],
                "x2": roi["x2"], "y2": roi["y2"]}
    return None


def apply_roi(raw_frame, roi):
    """Returns the frame to run detection on. For a polygon, everything
    outside the shape is blacked out (full frame size preserved). For a
    rectangle, the frame is cropped down to that box."""
    if roi is None:
        return raw_frame
    if roi["type"] == "rect":
        return raw_frame[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
    mask = np.zeros(raw_frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi["points"]], 255)
    return cv2.bitwise_and(raw_frame, raw_frame, mask=mask)


def _build_timeline(possession_log, buckets=TIMELINE_BUCKETS):
    """Compress the per-frame possession log into a small number of time
    segments, so the front end can draw a swing chart without shipping
    thousands of rows to the browser. Each bucket reports what share of its
    *contested* frames each team held."""
    if not possession_log:
        return []

    total = len(possession_log)
    size = max(1, -(-total // buckets))   # ceil, so we never exceed `buckets`
    out = []
    for start in range(0, total, size):
        chunk = possession_log[start:start + size]
        held = [r["team"] for r in chunk if r["team"] in (1, 2)]
        if held:
            a = sum(1 for t in held if t == 1) / len(held) * 100
        else:
            a = None  # nothing confidently tracked in this stretch
        out.append({
            "t": round(chunk[0]["time_s"], 1),
            "a": None if a is None else round(a, 1),
        })
    return out


# ---------------------------------------------------------------------------
# Goal-candidate heuristic
#
# There's no reliable way to actually detect a goal here: that normally needs
# a fixed camera on the goal line, and this camera pans to follow the ball —
# the exact same problem that made a static pitch-boundary mask unreliable.
#
# What we CAN do: flag moments where the ball moves unusually fast for THIS
# match, then disappears from tracking shortly after — consistent with a shot
# that went into the net, went out of play, or was followed by players
# celebrating (which disrupts tracking). This turns "scrub the whole match"
# into "review a handful of flagged moments" — a time-saver for manual
# tagging, not a replacement for it. Expect false positives (a hard clearance
# upfield) and misses (a slow, scrappy goal from a goalmouth scramble).
# ---------------------------------------------------------------------------
GOAL_MAX_GAP_FOR_SPEED_S = 0.6   # only compare positions this close together in time
GOAL_MIN_SPEED_SAMPLES = 8       # need a baseline of the match's own ball movement first
GOAL_SPEED_PERCENTILE = 90       # "fast" = faster than this % of the match's own ball movements
GOAL_MIN_SPEED_FRAC = 0.12       # frame-diagonals/sec floor, so a jittery match doesn't flag everything
GOAL_LOST_GAP_S = 1.2            # ball must then go undetected for at least this long
GOAL_LOOKAHEAD_S = 2.5           # ...starting within this long after the fast movement
GOAL_DEDUPE_WINDOW_S = 8.0       # merge candidates this close together in time
GOAL_MAX_CANDIDATES = 15


def _detect_goal_candidates(ball_track, frame_w, frame_h):
    """ball_track: [(time_s, x, y)] with x=y=None on frames the ball wasn't
    found, in time order. Returns candidate moments worth a manual look."""
    diag = (frame_w ** 2 + frame_h ** 2) ** 0.5 or 1
    detected = [(t, x, y) for t, x, y in ball_track if x is not None]
    if len(detected) < GOAL_MIN_SPEED_SAMPLES:
        return []

    speeds = []  # (time_of_later_point, speed_as_fraction_of_frame_diagonal_per_sec)
    for (t0, x0, y0), (t1, x1, y1) in zip(detected, detected[1:]):
        dt = t1 - t0
        if dt <= 0 or dt > GOAL_MAX_GAP_FOR_SPEED_S:
            continue
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        speeds.append((t1, (dist / diag) / dt))

    if len(speeds) < GOAL_MIN_SPEED_SAMPLES:
        return []

    threshold = max(np.percentile([s for _, s in speeds], GOAL_SPEED_PERCENTILE),
                     GOAL_MIN_SPEED_FRAC)
    fast_moments = [t for t, s in speeds if s >= threshold]

    candidates = []
    for t in fast_moments:
        after = [d[0] for d in detected if d[0] > t]
        if not after:
            continue
        nxt = after[0]
        if nxt - t >= GOAL_LOST_GAP_S:
            candidates.append(t)
            continue
        idx = next(i for i, d in enumerate(detected) if d[0] == nxt)
        for (ta, _, _), (tb, _, _) in zip(detected[idx:], detected[idx + 1:]):
            if tb - ta >= GOAL_LOST_GAP_S and tb <= t + GOAL_LOOKAHEAD_S + GOAL_LOST_GAP_S:
                candidates.append(t)
                break

    candidates.sort()
    deduped = []
    for t in candidates:
        if not deduped or t - deduped[-1] >= GOAL_DEDUPE_WINDOW_S:
            deduped.append(t)

    return [{"t": round(t, 1)} for t in deduped[:GOAL_MAX_CANDIDATES]]


def _transcode_for_browser(src, dst, log=print):
    """OpenCV writes mp4v, which many browsers refuse to play. If ffmpeg is
    available, re-encode to H.264 so the video works in the web player.
    Falls back to the original file if ffmpeg isn't installed."""
    if shutil.which("ffmpeg") is None:
        log("ffmpeg not found — keeping mp4v video (may not play in browser).")
        shutil.move(src, dst)
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
            check=True, capture_output=True,
        )
        os.remove(src)
        log("Re-encoded video for browser playback.")
        return True
    except subprocess.CalledProcessError as e:
        log(f"ffmpeg failed, keeping original video. {e.stderr.decode()[:200]}")
        shutil.move(src, dst)
        return False


def run_analysis(video_path, model_name="yolo26m.pt", conf=0.15, imgsz=1280,
                 roi_path="roi.json", label=None, progress_cb=None, log=print):
    """Runs the full pipeline on one video.

    progress_cb(done_frames, total_frames, ball_detections) is called
    periodically so a UI can show progress on a long job.

    Returns the result dict (also saved as result.json in the run folder).
    """
    def report(done, total, balls):
        if progress_cb:
            progress_cb(done, total, balls)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    log(f"Model: {model_name} | Confidence: {conf} | Detection size: {imgsz}")

    model = YOLO(model_name)
    person_class_ids, ball_class_ids = detect_relevant_classes(model, log=log)
    all_relevant_classes = person_class_ids + ball_class_ids
    team_assigner = TeamAssigner()
    roi = load_roi(roi_path, log=log)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Decide team colours early enough that a short clip still gets results:
    # a quarter of the way in, capped at the normal deadline.
    fallback_at = min(300, max(20, total_frames // 4)) if total_frames else 300

    if roi and roi["type"] == "rect":
        out_w, out_h = roi["x2"] - roi["x1"], roi["y2"] - roi["y1"]
    else:
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_video = os.path.join(run_dir, "_raw.mp4")
    out = cv2.VideoWriter(raw_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    possession_log = []
    frame_num = 0
    teams_initialized = False
    team_counts = {1: 0, 2: 0}
    held_frames = 0
    ball_detections = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame_in = apply_roi(raw_frame, roi)

        result = model.track(
            frame_in,
            classes=all_relevant_classes,
            tracker="bytetrack_custom.yaml",
            persist=True,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
        )[0]

        frame = frame_in.copy()
        boxes = result.boxes

        player_boxes = []
        ball_box = None

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            xyxys = boxes.xyxy.cpu().numpy()
            for track_id, cls, xyxy in zip(ids, clss, xyxys):
                if int(cls) in person_class_ids:
                    player_boxes.append((int(track_id), xyxy))
                elif int(cls) in ball_class_ids:
                    ball_box = xyxy

        if ball_box is not None:
            ball_detections += 1

        for track_id, box in player_boxes:
            team_assigner.collect_sample(frame_in, box, track_id)
        if not teams_initialized:
            teams_initialized = team_assigner.try_finalize_teams(frame_num, fallback_at)

        possessing_team = 0
        ball_xy = (None, None)
        if ball_box is not None:
            ball_center = bbox_center(ball_box)
            ball_xy = (round(float(ball_center[0]), 1), round(float(ball_center[1]), 1))
            best_dist = None
            best_team = None

            for track_id, box in player_boxes:
                team = team_assigner.get_player_team(frame, box, track_id) if teams_initialized else None
                if team is None:
                    continue
                color = team_assigner.team_color_bgr(team, FALLBACK_TEAM_BGR[team])
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID{track_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                dist = np.linalg.norm(bbox_center(box) - ball_center)
                if dist <= POSSESSION_MAX_DIST and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_team = team

            if best_team is not None:
                possessing_team = best_team

            bx1, by1, bx2, by2 = [int(v) for v in ball_box]
            cv2.circle(frame, (int((bx1 + bx2) / 2), int((by1 + by2) / 2)), 6, (0, 255, 255), -1)
        else:
            for track_id, box in player_boxes:
                team = team_assigner.get_player_team(frame, box, track_id) if teams_initialized else None
                if team is None:
                    continue
                color = team_assigner.team_color_bgr(team, FALLBACK_TEAM_BGR[team])
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        possession_log.append({
            "frame": frame_num,
            "time_s": round(frame_num / fps, 2),
            "team": possessing_team,
            "ball_x": ball_xy[0],
            "ball_y": ball_xy[1],
        })

        if possessing_team in (1, 2):
            team_counts[possessing_team] += 1
            held_frames += 1
        pct1 = team_counts[1] / held_frames * 100 if held_frames else 0
        pct2 = team_counts[2] / held_frames * 100 if held_frames else 0
        cv2.putText(frame, f"Team A poss: {pct1:.1f}%", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    team_assigner.team_color_bgr(1, FALLBACK_TEAM_BGR[1]), 2)
        cv2.putText(frame, f"Team B poss: {pct2:.1f}%", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    team_assigner.team_color_bgr(2, FALLBACK_TEAM_BGR[2]), 2)

        out.write(frame)
        frame_num += 1

        if frame_num % 25 == 0:
            report(frame_num, total_frames, ball_detections)

    cap.release()
    out.release()
    report(frame_num, total_frames or frame_num, ball_detections)

    df = pd.DataFrame(possession_log)
    df.to_csv(os.path.join(run_dir, "possession.csv"), index=False)

    _transcode_for_browser(raw_video, os.path.join(run_dir, "annotated.mp4"), log=log)

    ball_track = [(r["time_s"], r["ball_x"], r["ball_y"]) for r in possession_log]
    goal_candidates = _detect_goal_candidates(ball_track, out_w, out_h)
    if goal_candidates:
        log(f"Flagged {len(goal_candidates)} moment(s) worth checking for a goal.")

    held = df[df["team"] != 0] if len(df) else df
    result = {
        "id": run_id,
        "label": label or os.path.splitext(os.path.basename(video_path))[0],
        "created": datetime.now().isoformat(timespec="seconds"),
        "video": os.path.basename(video_path),
        "model": model_name,
        "conf": conf,
        "imgsz": imgsz,
        "fps": round(fps, 2),
        "frames": int(frame_num),
        "duration_s": round(frame_num / fps, 1) if fps else 0,
        "ball_detections": int(ball_detections),
        "ball_rate": round(ball_detections / max(frame_num, 1) * 100, 1),
        "contested_frames": int(len(held)),
        "possession_a": round((held["team"] == 1).mean() * 100, 1) if len(held) else None,
        "possession_b": round((held["team"] == 2).mean() * 100, 1) if len(held) else None,
        "color_a": team_assigner.team_color_hex(1),
        "color_b": team_assigner.team_color_hex(2),
        "timeline": _build_timeline(possession_log),
        "goal_candidates": goal_candidates,
        "goals": {"a": 0, "b": 0},
    }

    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    log(f"Done. Results saved to {run_dir}")
    return result


def list_runs():
    """All past runs, newest first."""
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for run_id in sorted(os.listdir(RUNS_DIR), reverse=True):
        path = os.path.join(RUNS_DIR, run_id, "result.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    out.append(json.load(f))
            except json.JSONDecodeError:
                continue
    return out


def load_run(run_id):
    path = os.path.join(RUNS_DIR, run_id, "result.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_run(result):
    path = os.path.join(RUNS_DIR, result["id"], "result.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)