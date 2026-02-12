"""Background worker thread for batch video transcription."""

import json
import os
import threading
from datetime import datetime, timezone

from models import Video, db
from transcription import (
    extract_audio,
    get_video_duration,
    segments_to_text,
    transcribe_audio,
    write_srt,
    write_txt,
)

_lock = threading.Lock()
_running = False


def _process_videos(app):
    """Worker loop: process pending videos one at a time."""
    global _running
    with app.app_context():
        while True:
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

                # Build output paths
                output_dir = os.path.join(app.root_path, "output")
                os.makedirs(output_dir, exist_ok=True)
                base = os.path.splitext(video.filename)[0]
                txt_path = os.path.join(output_dir, f"{base}_transcript.txt")
                srt_path = os.path.join(output_dir, f"{base}_transcript.srt")

                # Handle duplicate filenames by appending the video id
                if os.path.exists(txt_path) or os.path.exists(srt_path):
                    txt_path = os.path.join(output_dir, f"{base}_{video.id}_transcript.txt")
                    srt_path = os.path.join(output_dir, f"{base}_{video.id}_transcript.srt")

                write_txt(seg_dicts, txt_path)
                write_srt(seg_dicts, srt_path)

                # Update record
                full_text = segments_to_text(seg_dicts)
                video.transcript_text = full_text
                video.transcript_preview = full_text[:200]
                video.segments_json = json.dumps(seg_dicts)
                video.txt_path = txt_path
                video.srt_path = srt_path
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
    t = threading.Thread(target=_process_videos, args=(app,), daemon=True)
    t.start()
