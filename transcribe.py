"""Video transcription tool using OpenAI Whisper API."""

import argparse
import os
import subprocess
import sys
import tempfile

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def extract_audio(video_path: str) -> str:
    """Extract audio from video file as a temporary mp3."""
    fd, audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
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
        print("Error: FFmpeg not found. Install it and ensure it is on your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        os.unlink(audio_path)
        print(f"Error extracting audio:\n{exc.stderr.decode()}")
        sys.exit(1)
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


def write_txt(segments, output_path: str) -> None:
    """Write plain-text transcript."""
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg["text"].strip() + "\n")
    print(f"Transcript saved to {output_path}")


def write_srt(segments, output_path: str) -> None:
    """Write SRT subtitle file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"Subtitles saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe a video file using OpenAI Whisper.")
    parser.add_argument("video", help="Path to the video file")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"Error: File not found: {video_path}")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    base = os.path.splitext(video_path)[0]
    txt_path = f"{base}_transcript.txt"
    srt_path = f"{base}_transcript.srt"

    print(f"Extracting audio from {video_path}...")
    audio_path = extract_audio(video_path)

    try:
        print("Transcribing audio...")
        segments = transcribe_audio(audio_path)
        write_txt(segments, txt_path)
        write_srt(segments, srt_path)
        print("Done.")
    finally:
        os.unlink(audio_path)


if __name__ == "__main__":
    main()
