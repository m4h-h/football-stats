"""
Pick your pitch's actual boundary (any shape, not just a rectangle) so the
analyzer ignores an adjacent pitch even when they overlap at an angle.

Usage:
    python select_roi_polygon.py "input_videos/match1.mp4"

Click points around your pitch, one by one, going around the boundary
(order matters — go around the edge in order, not randomly).
Press ENTER when done (needs at least 3 points), 'r' to reset and start over,
'q' to quit without saving.

The window is automatically scaled down to fit your screen if the video is
larger than that (clicks are converted back to full-resolution coordinates
automatically, so accuracy isn't affected).

Saves to roi.json, which main.py automatically uses.
"""
import sys
import json
import cv2
import numpy as np

MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

points = []          # points in DISPLAY (scaled) coordinates, for drawing
orig_points = []      # same points converted to ORIGINAL frame coordinates, for saving
frame_display = None
display_frame_base = None
scale = 1.0


def redraw():
    global frame_display
    frame_display = display_frame_base.copy()
    for i, p in enumerate(points):
        cv2.circle(frame_display, p, 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(frame_display, points[i - 1], p, (0, 255, 0), 2)
    if len(points) > 2:
        cv2.line(frame_display, points[-1], points[0], (0, 255, 0), 1)
    cv2.imshow("Click around your pitch boundary", frame_display)


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        orig_points.append((int(x / scale), int(y / scale)))
        redraw()


def main():
    global display_frame_base, scale

    if len(sys.argv) < 2:
        print('Usage: python select_roi_polygon.py "<path_to_video>"')
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read a frame from that video.")
        sys.exit(1)

    h, w = frame.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)  # never upscale
    if scale < 1.0:
        display_frame_base = cv2.resize(frame, (int(w * scale), int(h * scale)))
        print(f"Video is {w}x{h} — scaling display down to fit your screen "
              f"({int(w*scale)}x{int(h*scale)}). Your clicks are still recorded "
              f"at full resolution automatically.")
    else:
        display_frame_base = frame.copy()

    print("Click points going around the boundary of YOUR pitch (any number of points, 3+).")
    print("Press ENTER when done, 'r' to reset, 'q' to quit without saving.")

    cv2.namedWindow("Click around your pitch boundary")
    cv2.setMouseCallback("Click around your pitch boundary", on_click)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13:  # ENTER
            if len(points) >= 3:
                break
            else:
                print("Need at least 3 points.")
        elif key == ord("r"):
            points.clear()
            orig_points.clear()
            redraw()
        elif key == ord("q"):
            print("Cancelled, nothing saved.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    roi = {"polygon": [[x, y] for x, y in orig_points]}
    with open("roi.json", "w") as f:
        json.dump(roi, f, indent=2)

    print(f"Saved {len(orig_points)}-point pitch boundary to roi.json")
    print("main.py will automatically use this from now on.")
    print("Delete roi.json anytime to go back to analyzing the full frame.")


if __name__ == "__main__":
    main()