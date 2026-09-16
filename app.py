"""
Local web app for running match analysis and browsing past results.

Start it with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.

Analysis and YouTube downloads each run in their own background thread so
the page stays responsive during a long job — the browser polls
/api/job and /api/download/job for progress.
"""
import os
import re
import shutil
import threading
import traceback

from flask import Flask, jsonify, request, send_from_directory, render_template
from werkzeug.utils import secure_filename

import analyzer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4GB — a full match at reasonable quality

INPUT_DIR = "input_videos"
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

# ---------------------------------------------------------------------------
# Analysis job (one at a time — this machine can't usefully do two at once)
# ---------------------------------------------------------------------------
_job_lock = threading.Lock()
_job = {
    "state": "idle",      # idle | running | done | error
    "label": None,
    "done": 0,
    "total": 0,
    "balls": 0,
    "lines": [],
    "run_id": None,
    "error": None,
}


def _set(**kwargs):
    with _job_lock:
        _job.update(kwargs)


def _log(line):
    with _job_lock:
        _job["lines"].append(str(line))
        del _job["lines"][:-40]


def _progress(done, total, balls):
    _set(done=done, total=total, balls=balls)


def _worker(video_path, model, conf, imgsz, roi_path, label):
    try:
        result = analyzer.run_analysis(
            video_path, model_name=model, conf=conf, imgsz=imgsz,
            roi_path=roi_path, label=label,
            progress_cb=_progress, log=_log,
        )
        _set(state="done", run_id=result["id"])
    except Exception as e:
        traceback.print_exc()
        _log(f"Failed: {e}")
        _set(state="error", error=str(e))


# ---------------------------------------------------------------------------
# YouTube download job (separate from analysis — this is network-bound, not
# CPU-bound, so there's no real reason to block it behind an analysis run)
# ---------------------------------------------------------------------------
_dl_lock = threading.Lock()
_dl_job = {
    "state": "idle",      # idle | running | done | error
    "url": None,
    "percent": 0.0,
    "speed": None,
    "eta": None,
    "filename": None,
    "error": None,
}


def _dl_set(**kwargs):
    with _dl_lock:
        _dl_job.update(kwargs)


def _dl_worker(url):
    try:
        import yt_dlp
    except ImportError:
        _dl_set(state="error", error="yt-dlp isn't installed. Run: pip install -r requirements.txt")
        return

    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes", 0)
            pct = round(done / total * 100, 1) if total else 0.0
            _dl_set(state="running", percent=pct,
                    speed=d.get("speed"), eta=d.get("eta"))
        elif d.get("status") == "finished":
            _dl_set(percent=100.0)

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(INPUT_DIR, "%(title).100s.%(ext)s"),
        "merge_output_format": "mp4",
        "progress_hooks": [hook],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        os.makedirs(INPUT_DIR, exist_ok=True)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
            base, _ = os.path.splitext(filepath)
            mp4_path = base + ".mp4"
            final_path = mp4_path if os.path.exists(mp4_path) else filepath
        _dl_set(state="done", filename=os.path.basename(final_path), percent=100.0)
    except Exception as e:
        traceback.print_exc()
        _dl_set(state="error", error=str(e))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Video library: list, upload a file, or fetch from YouTube
# ---------------------------------------------------------------------------
@app.route("/api/videos")
def api_videos():
    os.makedirs(INPUT_DIR, exist_ok=True)
    files = []
    for name in sorted(os.listdir(INPUT_DIR)):
        if name.lower().endswith(VIDEO_EXTS):
            path = os.path.join(INPUT_DIR, name)
            files.append({"name": name, "size_mb": round(os.path.getsize(path) / 1e6, 1)})
    return jsonify({"videos": files, "has_roi": os.path.exists("roi.json")})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify({"error": "No file received."}), 400

    name = secure_filename(f.filename)
    if not name.lower().endswith(VIDEO_EXTS):
        return jsonify({"error": f"Not a recognized video file ({', '.join(VIDEO_EXTS)})."}), 400

    os.makedirs(INPUT_DIR, exist_ok=True)
    dest = os.path.join(INPUT_DIR, name)

    # Don't silently overwrite an existing file with the same name
    if os.path.exists(dest):
        base, ext = os.path.splitext(name)
        n = 2
        while os.path.exists(os.path.join(INPUT_DIR, f"{base} ({n}){ext}")):
            n += 1
        name = f"{base} ({n}){ext}"
        dest = os.path.join(INPUT_DIR, name)

    f.save(dest)  # streamed to disk, not held fully in memory
    return jsonify({"ok": True, "name": name})


_YT_URL_RE = re.compile(r"^https?://(www\.)?(youtube\.com|youtu\.be)/")


@app.route("/api/download", methods=["POST"])
def api_download():
    with _dl_lock:
        if _dl_job["state"] == "running":
            return jsonify({"error": "A download is already in progress."}), 409

    data = request.get_json(force=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Paste a YouTube link first."}), 400
    if not _YT_URL_RE.match(url):
        return jsonify({"error": "That doesn't look like a YouTube link."}), 400

    _dl_set(state="running", url=url, percent=0.0, speed=None, eta=None,
            filename=None, error=None)
    threading.Thread(target=_dl_worker, args=(url,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/download/job")
def api_download_job():
    with _dl_lock:
        return jsonify(dict(_dl_job))


# ---------------------------------------------------------------------------
# Runs (past analyses)
# ---------------------------------------------------------------------------
@app.route("/api/runs")
def api_runs():
    return jsonify(analyzer.list_runs())


@app.route("/api/runs/<run_id>")
def api_run(run_id):
    result = analyzer.load_run(run_id)
    if result is None:
        return jsonify({"error": "No such run"}), 404
    return jsonify(result)


@app.route("/api/runs/<run_id>/goals", methods=["POST"])
def api_goals(run_id):
    result = analyzer.load_run(run_id)
    if result is None:
        return jsonify({"error": "No such run"}), 404
    data = request.get_json(force=True) or {}
    result["goals"] = {
        "a": max(0, int(data.get("a", 0))),
        "b": max(0, int(data.get("b", 0))),
    }
    analyzer.save_run(result)
    return jsonify(result)


@app.route("/api/runs/<run_id>", methods=["DELETE"])
def api_delete_run(run_id):
    path = os.path.join(analyzer.RUNS_DIR, run_id)
    if os.path.isdir(path):
        shutil.rmtree(path)
        return jsonify({"ok": True})
    return jsonify({"error": "No such run"}), 404


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    with _job_lock:
        if _job["state"] == "running":
            return jsonify({"error": "An analysis is already running."}), 409

    data = request.get_json(force=True) or {}
    video = data.get("video")
    if not video:
        return jsonify({"error": "Pick a video first."}), 400

    video_path = os.path.join(INPUT_DIR, video)
    if not os.path.exists(video_path):
        return jsonify({"error": f"Can't find {video}"}), 400

    model = data.get("model", "yolo26m.pt")
    conf = float(data.get("conf", 0.15))
    imgsz = int(data.get("imgsz", 1280))
    label = (data.get("label") or "").strip() or None
    roi_path = None  # pitch-limiting removed — see project notes

    _set(state="running", label=label or video, done=0, total=0, balls=0,
         lines=[], run_id=None, error=None)

    threading.Thread(
        target=_worker,
        args=(video_path, model, conf, imgsz, roi_path, label),
        daemon=True,
    ).start()

    return jsonify({"ok": True})


@app.route("/api/job")
def api_job():
    with _job_lock:
        return jsonify(dict(_job))


@app.route("/media/<run_id>/<path:filename>")
def media(run_id, filename):
    directory = os.path.abspath(os.path.join(analyzer.RUNS_DIR, run_id))
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(analyzer.RUNS_DIR, exist_ok=True)
    print("\n  Match analyzer running at http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)