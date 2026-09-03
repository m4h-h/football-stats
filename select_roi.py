"""
Pick the area of the frame that's actually your pitch, so the analyzer
ignores players on an adjacent pitch, spectators, etc.

Usage:
    python select_roi.py "input_videos/match1.mp4"

Opens the first frame in a window. Drag a rectangle around your pitch,
then press ENTER or SPACE to confirm (press 'c' to cancel and redo).

Saves the crop box to roi.json, which main.py will automatically use.
"""
import sys
import json
import cv2


def main():
    if len(sys.argv) < 2:
        print('Usage: python select_roi.py "<path_to_video>"')
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read a frame from that video.")
        sys.exit(1)

    print("Drag a rectangle around YOUR pitch (ignore any other pitch/area).")
    print("Press ENTER or SPACE when done, 'c' to cancel and redo.")

    x, y, w, h = cv2.selectROI("Select your pitch area", frame, showCrosshair=True)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        print("No area selected, nothing saved.")
        sys.exit(1)

    roi = {"x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h)}
    with open("roi.json", "w") as f:
        json.dump(roi, f, indent=2)

    print(f"Saved crop area to roi.json: {roi}")
    print("main.py will automatically use this from now on.")
    print("Delete roi.json anytime to go back to analyzing the full frame.")


if __name__ == "__main__":
    main()