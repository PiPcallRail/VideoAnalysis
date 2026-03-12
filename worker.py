"""Background worker thread for batch video transcription."""

import json
import os
import re
import threading
from datetime import datetime, timezone

from models import Video, db
from transcription import (
    analyze_screenshots,
    extract_audio,
    extract_frame,
    generate_report,
    generate_summary,
    get_video_duration,
    segments_to_text,
    transcribe_audio,
    write_srt,
    write_txt,
)

_lock = threading.Lock()
_running = False
_cancel_event = threading.Event()


def _sanitize_folder_name(name: str) -> str:
    """Create a filesystem-safe folder name from a video filename."""
    base = os.path.splitext(name)[0]
    return re.sub(r'[<>:"/\\|?*]', '_', base).strip('. ')


def _process_videos(app):
    """Worker loop: process pending videos one at a time."""
    global _running
    with app.app_context():
        while True:
            if _cancel_event.is_set():
                break

            video = (
                Video.query
                .filter_by(status="pending")
                .order_by(Video.created_at.asc())
                .first()
            )
            if video is None:
                break

            video.status = "processing"
            db.session.commit()

            audio_path = None
            try:
                # Get duration if missing
                if video.duration_seconds is None:
                    video.duration_seconds = get_video_duration(video.filepath)
                    db.session.commit()

                # Extract audio
                audio_path = extract_audio(video.filepath)

                # Transcribe
                segments = transcribe_audio(audio_path)

                # Convert segments to serialisable dicts
                seg_dicts = [dict(s) for s in segments]

                # Build per-video output folder
                output_base = os.path.join(app.root_path, "output")
                safe_name = _sanitize_folder_name(video.filename)
                video_output_dir = os.path.join(output_base, safe_name)

                # Handle duplicate folder names by appending video id
                if os.path.exists(video_output_dir):
                    marker_path = os.path.join(video_output_dir, ".video_id")
                    if os.path.isfile(marker_path):
                        with open(marker_path) as mf:
                            if mf.read().strip() != str(video.id):
                                video_output_dir = os.path.join(
                                    output_base, f"{safe_name}_{video.id}"
                                )
                    else:
                        video_output_dir = os.path.join(
                            output_base, f"{safe_name}_{video.id}"
                        )

                os.makedirs(video_output_dir, exist_ok=True)

                # Write marker file for re-processing detection
                with open(os.path.join(video_output_dir, ".video_id"), "w") as mf:
                    mf.write(str(video.id))

                screenshots_dir = os.path.join(video_output_dir, "screenshots")
                os.makedirs(screenshots_dir, exist_ok=True)

                # Write transcript files into the per-video folder
                txt_path = os.path.join(video_output_dir, "transcript.txt")
                srt_path = os.path.join(video_output_dir, "transcript.srt")
                write_txt(seg_dicts, txt_path)
                write_srt(seg_dicts, srt_path)

                # Screenshot analysis: scene detection + GPT-4o
                try:
                    screenshot_moments = analyze_screenshots(
                        seg_dicts, video_path=video.filepath
                    )
                except Exception:
                    screenshot_moments = []

                # Frame extraction
                for i, moment in enumerate(screenshot_moments):
                    if _cancel_event.is_set():
                        break
                    ts = moment["timestamp"]
                    desc_slug = re.sub(r'[^a-zA-Z0-9]+', '_', moment.get("description", ""))[:40]
                    img_filename = f"{i + 1:03d}_{ts:.0f}s_{desc_slug}.png"
                    img_path = os.path.join(screenshots_dir, img_filename)
                    try:
                        extract_frame(video.filepath, ts, img_path)
                        moment["image_path"] = img_path
                        moment["image_filename"] = img_filename
                    except Exception:
                        moment["image_path"] = None
                        moment["image_filename"] = None

                # Generate Word report
                try:
                    report_path = generate_report(
                        video.filename, seg_dicts, screenshot_moments, video_output_dir
                    )
                except Exception:
                    report_path = None

                # Update record
                full_text = segments_to_text(seg_dicts)
                video.transcript_text = full_text
                try:
                    video.transcript_preview = generate_summary(full_text)
                except Exception:
                    video.transcript_preview = full_text[:200]
                video.segments_json = json.dumps(seg_dicts)
                video.txt_path = txt_path
                video.srt_path = srt_path
                video.report_path = report_path
                video.screenshots_json = json.dumps([
                    {
                        "timestamp": m["timestamp"],
                        "description": m["description"],
                        "filename": m.get("image_filename"),
                    }
                    for m in screenshot_moments
                    if m.get("image_path")
                ])
                video.output_dir = video_output_dir
                video.status = "done"
                video.processed_at = datetime.now(timezone.utc)
                db.session.commit()

            except Exception as exc:
                video.status = "failed"
                video.error_message = str(exc)
                video.processed_at = datetime.now(timezone.utc)
                db.session.commit()

            finally:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)

    with _lock:
        _running = False


def start_processing(app):
    """Spawn the worker thread if it isn't already running."""
    global _running
    with _lock:
        if _running:
            return
        _running = True
    _cancel_event.clear()
    t = threading.Thread(target=_process_videos, args=(app,), daemon=True)
    t.start()


def stop_processing(app):
    """Signal the worker to stop after the current video finishes."""
    global _running
    _cancel_event.set()
    with app.app_context():
        Video.query.filter_by(status="processing").update({"status": "pending"})
        db.session.commit()
    with _lock:
        _running = False
