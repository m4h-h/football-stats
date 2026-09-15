"""
Command-line entry point. The actual pipeline lives in analyzer.py, which the
web app (app.py) also uses — so both stay in step automatically.

Usage:
    python main.py "input_videos/my_match.mp4" [model] [confidence] [imgsz]

Examples:
    python main.py "input_videos/match1.mp4"
    python main.py "input_videos/match1.mp4" yolo26m.pt 0.15 1280

If roi.json exists (from select_roi_polygon.py), everything outside that
shape is masked out before detection.

Results go to runs/<timestamp>/ — annotated.mp4, possession.csv, result.json.
Browse them with: python app.py
"""
import sys

import analyzer


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "<path_to_video>" [model] [confidence] [imgsz]')
        sys.exit(1)

    video_path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "yolo26m.pt"
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    imgsz = int(sys.argv[4]) if len(sys.argv) > 4 else 1280

    def progress(done, total, balls):
        if done % 100:
            return
        rate = balls / max(done, 1) * 100
        of = f"/{total}" if total else ""
        print(f"Processed {done}{of} frames... (ball found in {balls}, {rate:.1f}%)")

    result = analyzer.run_analysis(
        video_path, model_name=model, conf=conf, imgsz=imgsz,
        progress_cb=progress,
    )

    print("\n=== MATCH SUMMARY ===")
    print(f"Frames analyzed:   {result['frames']}")
    print(f"Ball found in:     {result['ball_detections']} frames ({result['ball_rate']}%)")
    if result["possession_a"] is not None:
        print(f"Team A possession: {result['possession_a']}%")
        print(f"Team B possession: {result['possession_b']}%")
    else:
        print("No clear possession detected — the ball wasn't tracked reliably.")
        print("Try a bigger model (yolo26l.pt) or a lower confidence (e.g. 0.1).")
    print(f"\nResults: runs/{result['id']}/")
    print("View them in the browser with: python app.py")


if __name__ == "__main__":
    main()