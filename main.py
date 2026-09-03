"""
Main pipeline: detect + track players/ball, assign teams, compute possession stats.

Usage:
    python main.py "input_videos/my_match.mp4" [model_name] [confidence] [imgsz]

Examples:
    python main.py "input_videos/match1.mp4"
    python main.py "input_videos/match1.mp4" yolov8s.pt
    python main.py "input_videos/match1.mp4" yolov8s.pt 0.15
    python main.py "input_videos/match1.mp4" yolov8s.pt 0.15 1280

If roi.json exists (created by select_roi_polygon.py), everything outside
that shape is masked out before detection — useful when multiple pitches
are visible and you only want to analyze your own game.

Outputs:
    output_videos/annotated.mp4   - video with team-colored boxes + possession HUD
    stats/possession.csv          - per-frame possession log
    Printed summary at the end.
"""
import sys
import os
import json
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from team_assigner import TeamAssigner

POSSESSION_MAX_DIST = 80  # pixels; ball must be within this of a player to count as "possessed"
TEAM_COLOR_BGR = {1: (0, 0, 255), 2: (255, 100, 0)}


def detect_relevant_classes(model):
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
            continue  # exclude referees — we don't want them team-classified
        if "ball" in lname:
            ball_ids.append(class_id)
        elif "person" in lname or "player" in lname or "goalkeeper" in lname:
            person_ids.append(class_id)

    if not person_ids or not ball_ids:
        raise RuntimeError(
            f"Couldn't find person/ball classes in this model's class list: {names}. "
            "Expected something like 'person'/'ball' or 'player'/'ball'."
        )

    print(f"Detected classes — players: {[names[i] for i in person_ids]}, "
          f"ball: {[names[i] for i in ball_ids]}")
    return person_ids, ball_ids


def bbox_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def load_roi():
    if os.path.exists("roi.json"):
        with open("roi.json") as f:
            roi = json.load(f)
        if "polygon" in roi:
            print(f"Using polygon pitch boundary from roi.json ({len(roi['polygon'])} points)")
            return {"type": "polygon", "points": np.array(roi["polygon"], dtype=np.int32)}
        else:
            print(f"Using rectangle crop area from roi.json: {roi}")
            return {"type": "rect", "x1": roi["x1"], "y1": roi["y1"], "x2": roi["x2"], "y2": roi["y2"]}
    return None


def apply_roi(raw_frame, roi):
    """Returns the frame to run detection on. For a polygon, everything
    outside the shape is blacked out (full frame size preserved). For a
    rectangle, the frame is cropped down to that box."""
    if roi is None:
        return raw_frame
    if roi["type"] == "rect":
        return raw_frame[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
    else:  # polygon
        mask = np.zeros(raw_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi["points"]], 255)
        masked = cv2.bitwise_and(raw_frame, raw_frame, mask=mask)
        return masked


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "<path_to_video>" [model_name] [confidence]')
        sys.exit(1)

    video_path = sys.argv[1]
    # yolov8s.pt (small) is a much better default than nano for spotting the ball —
    # still reasonably fast on CPU. Pass yolov8m.pt for even better accuracy if needed.
    model_name = sys.argv[2] if len(sys.argv) > 2 else "yolov8s.pt"
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15  # lower = catches more (incl. the ball), but more false positives
    imgsz = int(sys.argv[4]) if len(sys.argv) > 4 else 1280  # higher = better small-object detection (the ball), but slower

    os.makedirs("output_videos", exist_ok=True)
    os.makedirs("stats", exist_ok=True)

    print(f"Model: {model_name}  |  Confidence threshold: {conf}  |  Detection size: {imgsz}")

    model = YOLO(model_name)
    person_class_ids, ball_class_ids = detect_relevant_classes(model)
    all_relevant_classes = person_class_ids + ball_class_ids
    team_assigner = TeamAssigner()
    roi = load_roi()  # (x1, y1, x2, y2) or None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    if roi and roi["type"] == "rect":
        out_width, out_height = roi["x2"] - roi["x1"], roi["y2"] - roi["y1"]
    else:
        out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "output_videos/annotated.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, out_height),
    )

    possession_log = []  # rows: frame_num, time_s, team (0 = none)
    frame_num = 0
    teams_initialized = False
    team_counts = {1: 0, 2: 0}  # running tally, avoids rebuilding a DataFrame every frame
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

        player_boxes = []   # (track_id, xyxy)
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

        # Collect a color sample from every player seen so far (until teams are
        # finalized from a broad, robust pool instead of one noisy frame)
        for track_id, box in player_boxes:
            team_assigner.collect_sample(frame_in, box, track_id)
        if not teams_initialized:
            teams_initialized = team_assigner.try_finalize_teams(frame_num)

        # Determine each player's team + draw boxes
        possessing_team = 0
        if ball_box is not None:
            ball_center = bbox_center(ball_box)
            best_dist = None
            best_team = None

            for track_id, box in player_boxes:
                team = team_assigner.get_player_team(frame, box, track_id) if teams_initialized else None
                if team is None:
                    continue  # not a confident bib match — likely not one of your players, skip entirely
                color = TEAM_COLOR_BGR[team]
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
            # still draw player boxes even if ball not detected this frame
            for track_id, box in player_boxes:
                team = team_assigner.get_player_team(frame, box, track_id) if teams_initialized else None
                if team is None:
                    continue
                color = TEAM_COLOR_BGR[team]
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        possession_log.append({
            "frame": frame_num,
            "time_s": round(frame_num / fps, 2),
            "team": possessing_team,
        })

        # HUD (running tally — O(1) per frame instead of rebuilding a DataFrame)
        if possessing_team in (1, 2):
            team_counts[possessing_team] += 1
            held_frames += 1
        if held_frames > 0:
            pct1 = team_counts[1] / held_frames * 100
            pct2 = team_counts[2] / held_frames * 100
        else:
            pct1 = pct2 = 0
        cv2.putText(frame, f"Team A poss: {pct1:.1f}%", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEAM_COLOR_BGR[1], 2)
        cv2.putText(frame, f"Team B poss: {pct2:.1f}%", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEAM_COLOR_BGR[2], 2)

        out.write(frame)
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"Processed {frame_num} frames... (ball detected in {ball_detections}/{frame_num} so far, "
                  f"{ball_detections / frame_num * 100:.1f}%)")

    cap.release()
    out.release()

    df = pd.DataFrame(possession_log)
    df.to_csv("stats/possession.csv", index=False)

    held = df[df["team"] != 0]
    print("\n=== MATCH SUMMARY ===")
    print(f"Total frames analyzed: {len(df)}")
    print(f"Ball detected in {ball_detections}/{frame_num} frames ({ball_detections / max(frame_num,1) * 100:.1f}%)")
    if len(held) > 0:
        print(f"Team A possession: {(held['team'] == 1).mean() * 100:.1f}%")
        print(f"Team B possession: {(held['team'] == 2).mean() * 100:.1f}%")
    else:
        print("No clear possession detected — ball may not have been reliably tracked.")
        print("Try a bigger model (yolov8m.pt) or a lower confidence (e.g. 0.1).")
    print("\nAnnotated video: output_videos/annotated.mp4")
    print("Per-frame log:   stats/possession.csv")
    print("\nFor goals, run: python goal_logger.py \"" + video_path + "\"")


if __name__ == "__main__":
    main()