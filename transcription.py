"""Reusable transcription functions for the web app.

Adapted from transcribe.py — raises exceptions instead of calling sys.exit,
and avoids print() side-effects so callers control output.
"""

import glob
import json
import os
import shutil
import subprocess
import tempfile

from openai import OpenAI

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg",
}

# WinGet installs ffmpeg here; shutil.which() misses it when the shell hasn't reloaded PATH.
_WINGET_FFMPEG_GLOB = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "ffmpeg-*", "bin",
)


def _find_tool(name: str) -> str:
    """Return the path to *name* (e.g. 'ffmpeg'), searching PATH then WinGet."""
    found = shutil.which(name)
    if found:
        return found
    for bin_dir in glob.glob(_WINGET_FFMPEG_GLOB):
        candidate = os.path.join(bin_dir, f"{name}.exe")
        if os.path.isfile(candidate):
            return candidate
    return name  # fall back to bare name — will raise FileNotFoundError if missing


def scan_folder(path: str) -> list[str]:
    """Return sorted list of video file paths in *path*."""
    if not os.path.isdir(path):
        raise ValueError(f"Not a valid directory: {path}")
    results = []
    for entry in os.scandir(path):
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in VIDEO_EXTENSIONS:
            results.append(entry.path)
    results.sort()
    return results


def get_video_duration(path: str) -> float | None:
    """Use ffprobe to get video duration in seconds. Returns None on failure."""
    try:
        result = subprocess.run(
            [
                _find_tool("ffprobe"), "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def extract_audio(video_path: str) -> str:
    """Extract audio from video file as a temporary mp3.

    Raises RuntimeError on failure (instead of sys.exit).
    """
    fd, audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        subprocess.run(
            [
                _find_tool("ffmpeg"), "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "libmp3lame",
                "-q:a", "4",
                audio_path,
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        os.unlink(audio_path)
        raise RuntimeError("FFmpeg not found. Install it and ensure it is on your PATH.")
    except subprocess.CalledProcessError as exc:
        os.unlink(audio_path)
        raise RuntimeError(f"Error extracting audio: {exc.stderr.decode()}")
    return audio_path


def transcribe_audio(audio_path: str):
    """Send audio to OpenAI Whisper API and return segments."""
    client = OpenAI()
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )
    return response.segments


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def segments_to_text(segments) -> str:
    """Join segment text into a single plain-text string."""
    return "\n".join(seg["text"].strip() for seg in segments)


def write_txt(segments, output_path: str) -> None:
    """Write plain-text transcript (no print side-effects)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg["text"].strip() + "\n")


def write_srt(segments, output_path: str) -> None:
    """Write SRT subtitle file (no print side-effects)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
