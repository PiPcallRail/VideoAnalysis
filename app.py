"""Flask application for Video Transcriber."""

import json
import os
import re

from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for
from markupsafe import Markup

from werkzeug.utils import secure_filename

from models import Video, db
from transcription import VIDEO_EXTENSIONS, get_video_duration, scan_folder
from worker import start_processing

load_dotenv()


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///videos.db"
    db.init_app(app)

    with app.app_context():
        db.create_all()

    return app


app = create_app()


@app.route("/")
def index():
    videos = Video.query.order_by(Video.created_at.desc()).all()
    counts = {
        "done": Video.query.filter_by(status="done").count(),
        "processing": Video.query.filter_by(status="processing").count(),
        "pending": Video.query.filter_by(status="pending").count(),
        "failed": Video.query.filter_by(status="failed").count(),
    }
    return render_template("index.html", videos=videos, counts=counts)


@app.route("/scan", methods=["POST"])
def scan():
    folder = request.form.get("folder", "").strip()
    if not folder or not os.path.isdir(folder):
        flash("Invalid folder path.", "danger")
        return redirect(url_for("index"))

    try:
        paths = scan_folder(folder)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("index"))

    if not paths:
        flash("No video files found in that folder.", "warning")
        return redirect(url_for("index"))

    added = 0
    for filepath in paths:
        existing = Video.query.filter_by(filepath=filepath).first()
        if existing:
            continue
        video = Video(
            filename=os.path.basename(filepath),
            filepath=filepath,
            folder=folder,
            duration_seconds=get_video_duration(filepath),
            status="pending",
        )
        db.session.add(video)
        added += 1

    db.session.commit()

    if added:
        flash(f"Added {added} video(s) for processing.", "success")
        start_processing(app)
    else:
        flash("All videos in that folder are already in the database.", "info")

    return redirect(url_for("index"))


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        flash("No files selected.", "danger")
        return redirect(url_for("index"))

    upload_dir = os.path.join(app.root_path, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    added = 0
    for f in files:
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            continue

        safe_name = secure_filename(f.filename)
        dest = os.path.join(upload_dir, safe_name)

        # Avoid overwriting — append counter if file exists
        if os.path.exists(dest):
            base, ext_ = os.path.splitext(safe_name)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(upload_dir, f"{base}_{counter}{ext_}")
                counter += 1

        f.save(dest)

        existing = Video.query.filter_by(filepath=dest).first()
        if existing:
            continue

        video = Video(
            filename=os.path.basename(dest),
            filepath=dest,
            folder="uploads",
            duration_seconds=get_video_duration(dest),
            status="pending",
        )
        db.session.add(video)
        added += 1

    db.session.commit()

    if added:
        flash(f"Uploaded {added} video(s) for processing.", "success")
        start_processing(app)
    else:
        flash("No new video files were uploaded.", "warning")

    return redirect(url_for("index"))


@app.route("/video/<int:video_id>")
def detail(video_id):
    video = db.get_or_404(Video, video_id)
    segments = []
    if video.segments_json:
        raw = json.loads(video.segments_json)
        for idx, seg in enumerate(raw):
            segments.append({
                "index": idx,
                "timestamp": _format_timestamp(seg["start"]),
                "text": seg["text"],
            })
    return render_template("detail.html", video=video, segments=segments)


def _format_timestamp(seconds):
    """Convert float seconds to HH:MM:SS string."""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _highlight(text, query):
    """Return Markup with search term wrapped in highlight span."""
    from markupsafe import escape
    safe_text = str(escape(text))
    safe_query = str(escape(query))
    pattern = re.compile(re.escape(safe_query), re.IGNORECASE)
    result = pattern.sub(
        lambda m: f'<span class="search-highlight">{m.group()}</span>', safe_text
    )
    return Markup(result)


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    results = []
    total_matches = 0

    if query:
        videos = Video.query.filter_by(status="done").all()
        for video in videos:
            if not video.segments_json:
                continue
            segments = json.loads(video.segments_json)
            matches = []
            for idx, seg in enumerate(segments):
                if query.lower() in seg["text"].lower():
                    matches.append({
                        "index": idx,
                        "timestamp": _format_timestamp(seg["start"]),
                        "text": _highlight(seg["text"], query),
                    })
            if matches:
                results.append({"video": video, "matches": matches})
                total_matches += len(matches)

    return render_template(
        "search.html", query=query, results=results, total_matches=total_matches
    )


@app.route("/api/status")
def api_status():
    videos = Video.query.order_by(Video.created_at.desc()).all()
    counts = {
        "done": Video.query.filter_by(status="done").count(),
        "processing": Video.query.filter_by(status="processing").count(),
        "pending": Video.query.filter_by(status="pending").count(),
        "failed": Video.query.filter_by(status="failed").count(),
    }
    return jsonify(
        counts=counts,
        videos=[
            {
                "id": v.id,
                "filename": v.filename,
                "folder": v.folder,
                "duration_seconds": v.duration_seconds,
                "status": v.status,
                "transcript_preview": v.transcript_preview,
            }
            for v in videos
        ],
    )


@app.route("/download/<int:video_id>/<file_type>")
def download(video_id, file_type):
    video = db.get_or_404(Video, video_id)

    if file_type == "txt" and video.txt_path:
        return send_file(video.txt_path, as_attachment=True)
    elif file_type == "srt" and video.srt_path:
        return send_file(video.srt_path, as_attachment=True)
    else:
        flash("File not available.", "warning")
        return redirect(url_for("detail", video_id=video_id))


@app.route("/retry/<int:video_id>", methods=["POST"])
def retry(video_id):
    video = db.get_or_404(Video, video_id)
    if video.status == "failed":
        video.status = "pending"
        video.error_message = None
        video.processed_at = None
        db.session.commit()
        start_processing(app)
        flash(f"Retrying {video.filename}.", "info")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=5009)
