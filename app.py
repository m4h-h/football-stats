"""
Local web app for running match analysis and browsing past results.

Start it with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.

Analysis runs in a background thread so the page stays responsive during a
long job — the browser polls /api/job for progress.
"""
import os
import shutil
import threading
import traceback

from flask import Flask, jsonify, request, send_from_directory, render_template

import analyzer

app = Flask(__name__)

INPUT_DIR = "input_videos"
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

# Only one analysis runs at a time — this machine can't usefully do two at
# once anyway, and it keeps the progress model simple.
_job_lock = threading.Lock()
_job = {
    "state": "idle",      # idle | running | done | error
    "label": None,
    "done": 0,
    "total": 0,
    "balls": 0,
    "lines": [],          # log lines for the UI
    "run_id": None,
    "error": None,
}


def _set(**kwargs):
    with _job_lock:
        _job.update(kwargs)


def _log(line):
    with _job_lock:
        _job["lines"].append(str(line))
        del _job["lines"][:-40]  # keep the tail only


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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/videos")
def api_videos():
    """Videos available to analyze, from the input_videos folder."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    files = []
    for name in sorted(os.listdir(INPUT_DIR)):
        if name.lower().endswith(VIDEO_EXTS):
            path = os.path.join(INPUT_DIR, name)
            files.append({
                "name": name,
                "size_mb": round(os.path.getsize(path) / 1e6, 1),
            })
    return jsonify({
        "videos": files,
        "has_roi": os.path.exists("roi.json"),
    })


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
    """Goals are entered by hand — the camera angle can't support reliable
    automatic goal detection, so the scoreline is the one stat you tell it."""
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
    roi_path = "roi.json" if data.get("use_roi", True) else None

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
    """Serves the annotated video and CSV for a run."""
    directory = os.path.abspath(os.path.join(analyzer.RUNS_DIR, run_id))
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(analyzer.RUNS_DIR, exist_ok=True)
    print("\n  Match analyzer running at http://127.0.0.1:5000\n")
    # threaded=True so progress polling still works while analysis runs
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)