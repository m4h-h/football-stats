# Football Match Stats (7-a-side)

A simple pipeline to pull possession stats and goals out of your match videos,
built with YOLOv8 (detection + tracking), OpenCV, and KMeans team-color clustering.

This is a trimmed-down MVP of the full "Code In a Jiffy" football analysis tutorial —
just possession + manual goal tagging, no speed/distance yet (easy to add later).

## 1. Setup (run once, on your own machine — not in this chat)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first time you run `main.py`, `ultralytics` will auto-download the YOLOv8
weights file (`yolov8n.pt`, ~6MB) — needs an internet connection once.

## 2. Download your match video

```bash
python download_video.py "https://www.youtube.com/watch?v=XXXXXXXX"
```

This saves an mp4 into `input_videos/`.

(If you're filming your own matches instead, just drop the video file straight
into `input_videos/` and skip this step.)

## 3. Run the analysis

```bash
python main.py "input_videos/your_video.mp4"
```

This produces:
- `output_videos/annotated.mp4` — video with team-colored player boxes, ball marker,
  and a live possession % overlay
- `stats/possession.csv` — per-frame possession log
- A printed summary of overall possession %

## 4. Log goals (manual, separate pass)

Automatic goal detection needs a fixed goal-line camera to be reliable, so for
now this is a quick manual tagger — press a key when you see a goal:

```bash
python goal_logger.py "input_videos/your_video.mp4"
```

Controls: `a` = goal for Team A, `b` = goal for Team B, `space` = pause/resume,
`,` / `.` = step back/forward one frame while paused, `q` = quit and save.

Saves to `stats/goals.csv`.

## Notes & tuning

- **Team assignment** is based on jersey color clustering (KMeans), same idea
  as the tutorial. Works best with two visually distinct kits. Goalkeepers in
  a very different color may get misclassified — a known limitation, fine to
  ignore for casual stats.
- **Possession** is estimated by "which team's player is closest to the ball,
  within `POSSESSION_MAX_DIST` pixels" — tune that constant in `main.py` if
  possession looks off (lower video resolution = lower threshold).
- **Model size**: `yolov8n.pt` (nano) is fast but less accurate, especially on
  a small ball. If detection misses the ball a lot, try:
  `python main.py "input_videos/your_video.mp4" yolov8s.pt` (or `yolov8m.pt`)
  — slower but noticeably better ball detection.
- **7-a-side specifics**: smaller pitch and fewer players than the original
  11-a-side tutorial actually helps here — less occlusion, easier tracking.

## Natural next steps (not built yet)

- Speed/distance per player (needs perspective transform, like the tutorial)
- Pass counting (ball changing possession between same-team players)
- Auto-clip highlights around goal timestamps from `goals.csv`