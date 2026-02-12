"""SQLAlchemy Video model and database instance."""

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Video(db.Model):
    __tablename__ = "video"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.Text, nullable=False)
    filepath = db.Column(db.Text, nullable=False, unique=True)
    folder = db.Column(db.Text, nullable=False)
    duration_seconds = db.Column(db.Float, nullable=True)
    status = db.Column(db.Text, nullable=False, default="pending")
    error_message = db.Column(db.Text, nullable=True)
    transcript_text = db.Column(db.Text, nullable=True)
    transcript_preview = db.Column(db.Text, nullable=True)
    segments_json = db.Column(db.Text, nullable=True)
    txt_path = db.Column(db.Text, nullable=True)
    srt_path = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    processed_at = db.Column(db.DateTime, nullable=True)
