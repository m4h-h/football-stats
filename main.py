"""
Main pipeline: detect + track players/ball, assign teams, compute possession stats.

Usage:
    python main.py "input_videos/my_match.mp4"

Outputs:
    output_videos/annotated.mp4   - video with team-colored boxes + possession HUD
    stats/possession.csv          - per-frame possession log
    Printed summary at the end.
"""
import sys
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from team_assigner import TeamAssigner

PERSON_CLASS = 0
BALL_CLASS = 32          # "sports ball" in COCO classes
POSSESSION_MAX_DIST = 80  # pixels; ball must be within this of a player to count as "possessed"
TEAM_COLOR_BGR = {1: (0, 0, 255), 2: (255, 100, 0)}


def bbox_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "<path_to_video>"')
        sys.exit(1)

    video_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "yolov8n.pt"

    os.makedirs("output_videos", exist_ok=True)
    os.makedirs("stats", exist_ok=True)

    model = YOLO(model_name)
    team_assigner = TeamAssigner()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out = cv2.VideoWriter(
        "output_videos/annotated.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    possession_log = []  # rows: frame_num, time_s, team (0 = none)
    frame_num = 0
    teams_initialized = False

    results_gen = model.track(
        source=video_path,
        classes=[PERSON_CLASS, BALL_CLASS],
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )

    for result in results_gen:
        frame = result.orig_img.copy()
        boxes = result.boxes

        player_boxes = []   # (track_id, xyxy)
        ball_box = None

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            xyxys = boxes.xyxy.cpu().numpy()

            for track_id, cls, xyxy in zip(ids, clss, xyxys):
                if int(cls) == PERSON_CLASS:
                    player_boxes.append((int(track_id), xyxy))
                elif int(cls) == BALL_CLASS:
                    ball_box = xyxy

        # Initialize the two team colors once we see enough players in a frame
        if not teams_initialized and len(player_boxes) >= 4:
            ok = team_assigner.assign_team_colors(frame, [b for _, b in player_boxes])
            teams_initialized = ok

        # Determine each player's team + draw boxes
        possessing_team = 0
        if ball_box is not None:
            ball_center = bbox_center(ball_box)
            best_dist = None
            best_team = None

            for track_id, box in player_boxes:
                team = team_assigner.get_player_team(frame, box, track_id) if teams_initialized else None
                color = TEAM_COLOR_BGR.get(team, (200, 200, 200))
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID{track_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                if team is not None:
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
                color = TEAM_COLOR_BGR.get(team, (200, 200, 200))
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        possession_log.append({
            "frame": frame_num,
            "time_s": round(frame_num / fps, 2),
            "team": possessing_team,
        })

        # HUD
        df_so_far = pd.DataFrame(possession_log)
        held = df_so_far[df_so_far["team"] != 0]
        if len(held) > 0:
            pct1 = (held["team"] == 1).mean() * 100
            pct2 = (held["team"] == 2).mean() * 100
        else:
            pct1 = pct2 = 0
        cv2.putText(frame, f"Team A poss: {pct1:.1f}%", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEAM_COLOR_BGR[1], 2)
        cv2.putText(frame, f"Team B poss: {pct2:.1f}%", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEAM_COLOR_BGR[2], 2)

        out.write(frame)
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"Processed {frame_num} frames...")

    out.release()

    df = pd.DataFrame(possession_log)
    df.to_csv("stats/possession.csv", index=False)

    held = df[df["team"] != 0]
    print("\n=== MATCH SUMMARY ===")
    print(f"Total frames analyzed: {len(df)}")
    if len(held) > 0:
        print(f"Team A possession: {(held['team'] == 1).mean() * 100:.1f}%")
        print(f"Team B possession: {(held['team'] == 2).mean() * 100:.1f}%")
    else:
        print("No clear possession detected — ball may not have been reliably tracked.")
    print("\nAnnotated video: output_videos/annotated.mp4")
    print("Per-frame log:   stats/possession.csv")
    print("\nFor goals, run: python goal_logger.py \"" + video_path + "\"")


if __name__ == "__main__":
    main()