# Smart Attendance System

Face-recognition attendance for a classroom door: enroll each student from the webcam, match faces as they walk in, and keep **one attendance row per student per day** with the real first-seen time.

Built from the project blueprint for Vivek Das (B.Tech CSE, KIIT University).

## What it does

- **History** — student roster, attendance rows, and kiosk matches are stored in SQLite and mirrored in the browser (`localStorage`). Refreshing the page restores them, including on Vercel where the server disk is ephemeral.
- **Enrollment** — webcam captures 5–10 stills at different angles, stores 128-d face embeddings plus reference photos
- **Kiosk** — samples the camera every ~2 seconds, matches every face in the frame, optional blink/motion liveness
- **Logging** — SQLite `UNIQUE(student_id, date)` so a student who walks past five times still has one row
- **Present / Late** — inside the attendance window → Present; after it → Late, still with the true `time_in`
- **Close day** — writes `Absent` for everyone with no row that date
- **Teacher report** — daily roll, week/month %, class filter, CSV and PDF export

Recognition runs in the browser with [face-api.js](https://github.com/justadudewhohacks/face-api.js) (TinyFaceDetector + landmarks + recognition net). The FastAPI backend stores embeddings and enforces attendance rules.

## Run locally

You need Python 3.11+ and a webcam. Use Chrome or Edge. Camera access works on `http://127.0.0.1` and on HTTPS.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 43147 --reload
```

Open [http://127.0.0.1:43147](http://127.0.0.1:43147).

1. **Enroll faces** — allow the camera, fill student ID / name / class, capture at least five angles, save
2. **Camera kiosk** — Start scanning, stand in frame; first match of the day is logged
3. **Today’s roll** — filter by class, export CSV/PDF, mark remaining students Absent

## Tests

```bash
pytest -q
```

## Settings

Default window is 08:00–08:30 (`Asia/Kolkata`). After 08:30 a first detection is still stored, as **Late**. Tune this under **Window & settings**. Cosine match threshold starts at `0.58`.

Data lives in `data/attendance.db` and `data/photos/`. No cloud account is required.

## Stack

FastAPI, Jinja2, SQLite, face-api.js, ReportLab (PDF).
