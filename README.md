# Football Match Stats — 7-a-side Possession & Goals Analyzer

A Python tool that watches a video of your football match and automatically
works out **ball possession percentages** for each team, using AI object
detection — plus a simple manual tool for logging goals. Built to run on
your own laptop, on your own match footage.

This document explains what the project does, how it technically works, and
how to use it — written so that even if you've never coded before, you can
follow along.

---

## 1. What this actually does (plain-English overview)

You give it a video of a match. It then:

1. **Watches every frame** of the video and finds every player and the ball
   using an AI model (an "object detector").
2. **Follows each player** across frames, giving them a consistent ID number
   (like a name tag), even as they move around — this is called "tracking."
3. **Works out which team each player is on**, automatically, just by
   looking at the color of their bib/jersey (no manual tagging needed).
4. **Watches where the ball is relative to players** — whichever team has a
   player closest to the ball is counted as "in possession" for that
   instant.
5. Adds all of that up across the whole video to give you a **possession
   percentage** for each team, and produces a video with boxes drawn around
   players (colored by team) so you can visually check the results.
6. Separately, a small tool lets you **manually tag goals** by pressing a
   key while watching the video (automatic goal detection isn't reliable
   without a dedicated goal-line camera, so this is the practical approach).

---

## 2. The technical pipeline (how it works under the hood)

If you're curious about the "why" behind each piece, here's the full chain:

### Step 1 — Object detection (YOLO)
We use a pretrained AI model called **YOLOv8** ("You Only Look Once version
8"), made by a company called Ultralytics. YOLO looks at a single video
frame and draws a box around every object it recognizes, along with a label
(e.g. "person", "sports ball") and a confidence score (how sure it is).

We use the general-purpose pretrained version of this model (it wasn't
specifically trained on football footage — more on that in section 6).

### Step 2 — Tracking (ByteTrack)
Detection alone treats every frame independently — it doesn't know that the
player in frame 100 is the "same" player as in frame 101. **ByteTrack** (a
tracking algorithm bundled with YOLO) solves this: it looks at how boxes
move and overlap between consecutive frames and assigns a consistent ID
number to each object, so we can follow individuals over time.

We use a **custom-tuned version** of ByteTrack (`bytetrack_custom.yaml`)
with a longer "memory" (`track_buffer: 90` instead of the default 30) — this
means if a player is briefly hidden behind another player or blurred by fast
motion, the tracker is more likely to correctly recognize them as the same
person when they reappear, rather than assigning a brand new ID.

### Step 3 — Restricting to your pitch (ROI masking)
If your camera view includes an adjacent pitch or spectators, we don't want
those people detected as players. `select_roi_polygon.py` lets you click
around the actual boundary of your pitch once — this shape gets saved to
`roi.json`. During analysis, everything **outside** that shape is blacked
out before detection even runs, so people outside your pitch are invisible
to the AI model entirely.

### Step 4 — Team assignment (hue-based color clustering)
This is the cleverest part, so here's the full reasoning:

- For each player's bounding box, we crop out roughly the torso area (where
  a bib/jersey shows, avoiding the head and legs).
- We convert that crop from **RGB color** (red/green/blue) to **HSV color**
  (hue/saturation/value). The reason: RGB mixes color and brightness
  together, so the same red bib in sunlight vs. shadow can look like two
  completely different RGB colors. **Hue** captures just the color itself,
  independent of brightness — so a red bib is "red" whether it's in bright
  sun or deep shadow. This was a real problem we hit early on (players
  flickering between team colors) and switching to hue fixed it.
- We filter out pixels that are pitch-green, low-saturation (skin, shadow,
  white lines), or overexposed/underexposed — leaving mostly just
  bib-colored pixels.
- We collect one color sample from several different players (waiting for
  ~8 unique players, or falling back after 300 frames if the pitch is
  never that crowded), and run **KMeans clustering** (an algorithm that
  automatically groups similar data points — here, similar colors — into
  clusters) with 2 clusters. This finds the two bib colors without ever
  being told what they are.
- From then on, any player's color gets compared to both cluster centers.
  If it's close enough to one of them, that's their team. If it's not
  close to *either* (e.g. a spectator in a completely different color),
  they're rejected — not shown as a player at all.
- To avoid a single bad frame (motion blur, weird lighting) flipping a
  player's team color, we don't trust any single frame. Instead we keep a
  rolling history of the last 10 classifications for each player and go
  with whichever team "wins" the majority vote.

### Step 5 — Possession calculation
For every frame where the ball is detected, we find whichever tracked
player is closest to it (within a maximum pixel distance, so a far-away
player doesn't get credited). That player's team is counted as "in
possession" for that frame. Possession % is just: (frames team A had the
ball) ÷ (total frames where anyone clearly had the ball).

### Step 6 — Goal logging (manual)
Automatically detecting goals reliably needs a fixed camera looking straight
at the goal line — with a single moving match camera, it's not reliable
enough to trust. So `goal_logger.py` gives you a simple video player where
you press `a`/`b` to log a goal for each team as you watch, saved to a CSV.

---

## 3. Project files, explained

| File | What it does |
|---|---|
| `README.md` | This file. |
| `requirements.txt` | The list of Python packages this project needs. |
| `download_video.py` | Downloads a match video from a YouTube link. |
| `select_roi_polygon.py` | Lets you click around your pitch's boundary so other areas (other pitches, spectators) get ignored. **Use this one**, not `select_roi.py`. |
| `select_roi.py` | An earlier, simpler rectangle-only version of the pitch selector. Kept for reference — the polygon version is better for angled camera views. |
| `team_assigner.py` | The hue-based color clustering logic (Step 4 above) that figures out which team each player is on. |
| `bytetrack_custom.yaml` | Settings for the player-tracking algorithm, tuned to hold onto the same player ID longer. |
| `main.py` | The main script — runs the whole pipeline end-to-end and produces your results. |
| `goal_logger.py` | The manual goal-tagging tool. |
| `train_custom_model.py` | A script to fine-tune YOLO on a football-specific dataset instead of the generic pretrained one. **See section 6 below before using this** — in testing, this made results worse, not better, due to a mismatch between the training data and this kind of match footage. |

---

## 4. Setup (do this once)

You'll need [Python](https://python.org) and [Homebrew](https://brew.sh)
installed on your Mac.

```bash
# 1. Install ffmpeg (needed to download/trim videos)
brew install ffmpeg

# 2. Create a virtual environment (an isolated space for this project's
#    Python packages, so they don't clash with anything else on your system)
python3 -m venv venv

# 3. Activate it — you'll need to do this every time you open a new
#    terminal window to work on this project. You'll know it worked
#    because your terminal prompt will show (venv) at the start.
source venv/bin/activate

# 4. Install all the required packages into this virtual environment
pip install -r requirements.txt
```

**Important:** keep this project folder somewhere that ISN'T synced by
iCloud Drive (e.g. not directly on your Desktop, if you have "Desktop &
Documents Folders" syncing turned on in iCloud settings). Cloud-synced
folders can turn simple file operations into extremely slow cloud
downloads, since Python environments contain thousands of small files.

The first time you run `main.py`, it will also auto-download the YOLO model
weights file (a few megabytes) — this needs an internet connection once.

---

## 5. Step-by-step usage

### Step A — Get your match video

If it's on YouTube:
```bash
python download_video.py "https://www.youtube.com/watch?v=XXXXXXXX"
```
This saves an `.mp4` into `input_videos/`. If you already have a video file,
just place it directly in `input_videos/` instead — no need to download.

**Tip:** rename your video file to something simple (no spaces or special
characters), e.g. `match1.mp4`, using:
```bash
mv "input_videos/original weird name.mp4" "input_videos/match1.mp4"
```
This avoids a lot of headaches with quoting filenames in later commands.

### Step B — (Recommended) trim a short test clip first

Before running on a full 60-90 minute match (which can take hours), test on
a short clip first:
```bash
ffmpeg -i "input_videos/match1.mp4" -ss 00:02:00 -t 00:02:00 -c copy "input_videos/test_clip.mp4"
```
This grabs a 2-minute clip starting at the 2:00 mark.

### Step C — Mark out your pitch

If your camera view includes anything you don't want analyzed (another
pitch, spectators, parked cars, etc.):
```bash
python select_roi_polygon.py "input_videos/test_clip.mp4"
```
A window opens showing a frame from your video. Click points one by one,
going around the actual boundary of your pitch. Press ENTER when done
(3+ points needed), `r` to restart if you misclick. This saves `roi.json`,
which `main.py` will automatically use from now on. Delete `roi.json`
anytime to go back to analyzing the full frame.

### Step D — Run the analysis

```bash
python -u main.py "input_videos/test_clip.mp4" yolov8s.pt 0.15 1280
```

Breaking down those arguments:
- `"input_videos/test_clip.mp4"` — your video file
- `yolov8s.pt` — which YOLO model to use (`yolov8n.pt` is faster but less
  accurate; `yolov8s.pt` is a good balance; `yolov8m.pt` is slower but more
  accurate again)
- `0.15` — confidence threshold (how sure the AI needs to be before it
  counts something as a detection — lower catches more, including more
  false positives)
- `1280` — detection resolution (higher helps catch small/fast objects like
  the ball, at the cost of speed)

This produces:
- `output_videos/annotated.mp4` — your video with colored boxes around
  players and a live possession % overlay
- `stats/possession.csv` — a per-frame log of who had the ball
- A printed summary in the terminal

### Step E — Log goals

```bash
python goal_logger.py "input_videos/test_clip.mp4"
```
A video window opens. Press `a` for a Team A goal, `b` for Team B, `space`
to pause/resume, `,`/`.` to step frame-by-frame while paused, `q` to save
and quit. Saves to `stats/goals.csv`.

---

## 6. Known limitations (and what we learned trying to fix them)

Being upfront about where this stands, since a lot of debugging went into
figuring these out:

- **Ball detection isn't perfect.** A football is small, fast, and easily
  confused with other round objects or motion blur. Expect somewhere around
  15-20% of frames to have a confirmed ball detection with the current
  setup — good enough for a rough possession estimate, not perfect
  frame-by-frame accuracy.
- **We tried training a custom model** specifically for football (using a
  public dataset with dedicated player/referee/ball labels, instead of the
  generic pretrained model). This actually made results *worse* on this
  footage — the public dataset was made of professional broadcast camera
  footage (close, zoomed-in angles), which doesn't match a wide-angle,
  elevated, amateur recording of a 7-a-side pitch. The model had learned
  what a football looks like on TV, not what it looks like from far away
  on a phone camera. **Lesson:** a custom-trained model is only better than
  a generic one if its training data actually resembles your footage. If
  you want to revisit this, the right approach would be labeling a batch of
  frames from **your own footage** (not a generic public dataset) and
  fine-tuning on that — a bigger time investment, but a domain match.
- **Team assignment depends on your two bib colors being visually
  distinct** (e.g. red vs. yellow works well). Very similar colors, or a
  referee's kit that happens to be close to one of your bib colors, may
  cause occasional misclassification.
- **The whole pitch needs to be in frame** for complete stats — if the
  camera doesn't capture the full pitch, anything happening off-screen
  simply isn't in the footage to analyze. For future recordings, filming
  from a higher, wider, or more central position will give more complete
  results.
- **This runs on CPU** (a normal laptop, no GPU) — processing is slow,
  roughly a few frames per second. A full match can take hours. Testing on
  short trimmed clips first is strongly recommended.

---

## 7. Possible next steps (not built yet)

- Player speed/distance covered (would need perspective transformation —
  converting pixel movement into real-world meters)
- Pass counting (detecting when the ball changes possession between
  same-team players)
- Auto-generating highlight clips around goal timestamps from `goals.csv`
- Fine-tuning a model on your own labeled footage, for better accuracy than
  either the generic pretrained model or the mismatched public dataset