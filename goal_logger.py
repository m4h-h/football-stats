"""
Simple manual goal tagger. Play through the video and press:
    a  -> log a goal for Team A
    b  -> log a goal for Team B
    space -> pause/resume
    ,  -> step back a few frames (while paused)
    .  -> step forward a few frames (while paused)
    q  -> quit and save

Usage:
    python goal_logger.py "input_videos/my_match.mp4"

Output:
    stats/goals.csv
"""
import sys
import os
import cv2
import pandas as pd

STEP = 5


def main():
    if len(sys.argv) < 2:
        print('Usage: python goal_logger.py "<path_to_video>"')
        sys.exit(1)

    video_path = sys.argv[1]
    os.makedirs("stats", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    goals = []
    paused = False

    print("Controls: [a]=goal Team A  [b]=goal Team B  [space]=pause/resume  "
          "[,]/[.]=step when paused  [q]=quit & save")

    frame_num = 0
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_num - 1, 0))

        time_s = frame_num / fps
        label = f"t={time_s:.1f}s  frame={frame_num}  goals_logged={len(goals)}"
        cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Goal Logger", frame)

        key = cv2.waitKey(0 if paused else 20) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("a"):
            goals.append({"team": "A", "frame": frame_num, "time_s": round(time_s, 1)})
            print(f"Goal logged: Team A at {time_s:.1f}s")
        elif key == ord("b"):
            goals.append({"team": "B", "frame": frame_num, "time_s": round(time_s, 1)})
            print(f"Goal logged: Team B at {time_s:.1f}s")
        elif paused and key == ord("."):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num + STEP)
        elif paused and key == ord(","):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_num - STEP, 0))

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(goals)
    df.to_csv("stats/goals.csv", index=False)
    print(f"\nSaved {len(goals)} goals to stats/goals.csv")
    if len(goals) > 0:
        print(df["team"].value_counts())


if __name__ == "__main__":
    main()