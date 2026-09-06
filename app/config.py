from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
# Vercel serverless functions can write only to /tmp.
if os.environ.get("VERCEL"):
    DATA_DIR = Path("/tmp/smart-attendance")
else:
    DATA_DIR = BASE_DIR / "data"
PHOTO_DIR = DATA_DIR / "photos"
DB_PATH = DATA_DIR / "attendance.db"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

TIMEZONE = "Asia/Kolkata"

# Cosine similarity: face-api.js 128-d descriptors typically match above ~0.55–0.65
MATCH_THRESHOLD = 0.58
# Euclidean distance fallback (face-api default is often 0.6)
EUCLIDEAN_THRESHOLD = 0.52

MIN_ENROLL_PHOTOS = 5
MAX_ENROLL_PHOTOS = 10

DEFAULT_SETTINGS = {
    "class_start": "08:00",
    "late_after": "08:30",
    "close_after": "18:00",
    "institution": "KIIT University",
    "match_threshold": str(MATCH_THRESHOLD),
}
