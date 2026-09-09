from __future__ import annotations

import base64
import csv
import io
import json
import re
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.attendance import daily_report, late_cutoff, mark_absents_for_date, parse_hhmm, range_report, record_attendance
from app.config import (
    EUCLIDEAN_THRESHOLD,
    MATCH_THRESHOLD,
    MAX_ENROLL_PHOTOS,
    MIN_ENROLL_PHOTOS,
    PHOTO_DIR,
    STATIC_DIR,
    TEMPLATE_DIR,
)
from app.db import (
    add_scan_event,
    class_sections,
    export_snapshot,
    get_db,
    get_settings,
    get_student,
    init_db,
    list_scan_events,
    list_students,
    load_all_embeddings,
    now,
    restore_snapshot,
    save_photo,
    set_setting,
    student_attendance_history,
    student_photo_paths,
    today_str,
)
from app.exports import attendance_pdf
from app.recognition import match_faces


ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,32}$")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


PHOTO_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Smart Attendance System", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/photos", StaticFiles(directory=PHOTO_DIR), name="photos")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def render(request: Request, name: str, **ctx: Any) -> HTMLResponse:
    with get_db() as conn:
        settings = get_settings(conn)
        sections = class_sections(conn)
        enrolled = conn.execute("SELECT COUNT(*) AS n FROM students").fetchone()["n"]
        present_today = conn.execute(
            "SELECT COUNT(*) AS n FROM attendance WHERE date = ? AND status IN ('Present', 'Late')",
            (today_str(),),
        ).fetchone()["n"]
    payload = {
        "settings": settings,
        "sections": sections,
        "today": today_str(),
        "clock": now().strftime("%d %b %Y · %H:%M"),
        "enrolled_count": enrolled,
        "present_today": present_today,
        "late_cutoff": late_cutoff(now(), settings).strftime("%H:%M"),
        **ctx,
    }
    return templates.TemplateResponse(request, name, payload)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, class_section: str | None = None, day: str | None = None):
    day = day or today_str()
    with get_db() as conn:
        rows = daily_report(conn, day, class_section)
        total = len(rows)
        present = sum(1 for r in rows if r["status"] == "Present")
        late = sum(1 for r in rows if r["status"] == "Late")
        absent = sum(1 for r in rows if r["status"] == "Absent")
        unmatched_enrolled = conn.execute(
            """
            SELECT COUNT(*) AS n FROM students s
            LEFT JOIN embeddings e ON e.student_id = s.student_id
            WHERE e.id IS NULL
            """
        ).fetchone()["n"]
    return render(
        request,
        "dashboard.html",
        active="dashboard",
        rows=rows,
        day=day,
        class_section=class_section or "",
        stats={
            "total": total,
            "present": present,
            "late": late,
            "absent": absent,
            "unmatched": unmatched_enrolled,
        },
    )


@app.get("/students", response_class=HTMLResponse)
def students_page(request: Request, class_section: str | None = None):
    with get_db() as conn:
        students = [dict(s) for s in list_students(conn, class_section)]
    for s in students:
        s["photos"] = student_photo_paths(s["student_id"])
    return render(
        request,
        "students.html",
        active="students",
        students=students,
        class_section=class_section or "",
    )


@app.get("/students/{student_id}", response_class=HTMLResponse)
def student_history_page(request: Request, student_id: str):
    sid = student_id.strip().upper()
    with get_db() as conn:
        student = get_student(conn, sid)
        if not student:
            raise HTTPException(404, "Student not found")
        history = student_attendance_history(conn, sid)
        events = [
            e
            for e in list_scan_events(conn, limit=200)
            if (e.get("student_id") or "").upper() == sid
        ]
    photos = student_photo_paths(sid)
    return render(
        request,
        "student_history.html",
        active="students",
        student=dict(student),
        history=history,
        events=events,
        photos=photos,
    )


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request):
    with get_db() as conn:
        events = list_scan_events(conn, limit=120)
        attendance = [
            dict(r)
            for r in conn.execute(
                """
                SELECT a.date, a.time_in, a.status, s.student_id, s.name, s.class_section
                FROM attendance a
                JOIN students s ON s.student_id = a.student_id
                ORDER BY a.date DESC, a.time_in DESC
                LIMIT 200
                """
            )
        ]
    return render(
        request,
        "history.html",
        active="history",
        events=events,
        attendance=attendance,
    )


@app.get("/enroll", response_class=HTMLResponse)
def enroll_page(request: Request):
    return render(request, "enroll.html", active="enroll")


@app.get("/kiosk", response_class=HTMLResponse)
def kiosk_page(request: Request):
    return render(request, "kiosk.html", active="kiosk")


@app.get("/reports", response_class=HTMLResponse)
def reports_page(
    request: Request,
    start: str | None = None,
    end: str | None = None,
    class_section: str | None = None,
):
    end = end or today_str()
    start = start or (date.fromisoformat(end) - timedelta(days=6)).isoformat()
    with get_db() as conn:
        rows = range_report(conn, start, end, class_section)
    return render(
        request,
        "reports.html",
        active="reports",
        rows=rows,
        start=start,
        end=end,
        class_section=class_section or "",
    )


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return render(request, "settings.html", active="settings")


@app.post("/settings")
def save_settings(
    class_start: str = Form(...),
    late_after: str = Form(...),
    close_after: str = Form(...),
    institution: str = Form(...),
    match_threshold: str = Form(...),
):
    with get_db() as conn:
        set_setting(conn, "class_start", class_start)
        try:
            if parse_hhmm(late_after) < parse_hhmm(class_start):
                late_after = class_start
        except ValueError:
            late_after = class_start
        set_setting(conn, "late_after", late_after)
        set_setting(conn, "close_after", close_after)
        set_setting(conn, "institution", institution.strip() or "KIIT University")
        set_setting(conn, "match_threshold", match_threshold)
    return RedirectResponse("/settings?saved=1", status_code=303)


class EmbeddingIn(BaseModel):
    descriptor: list[float] = Field(min_length=64, max_length=256)
    photo_jpeg_base64: str | None = None


class EnrollIn(BaseModel):
    student_id: str
    name: str
    class_section: str
    guardian_whatsapp_number: str | None = None
    embeddings: list[EmbeddingIn]


@app.post("/api/enroll")
def api_enroll(payload: EnrollIn):
    student_id = payload.student_id.strip().upper()
    if not ID_RE.match(student_id):
        raise HTTPException(400, "Student ID must be 2–32 letters, numbers, _ or -")
    name = payload.name.strip()
    section = payload.class_section.strip()
    if not name or not section:
        raise HTTPException(400, "Name and class/section are required")
    n = len(payload.embeddings)
    if n < MIN_ENROLL_PHOTOS or n > MAX_ENROLL_PHOTOS:
        raise HTTPException(
            400,
            f"Capture {MIN_ENROLL_PHOTOS}–{MAX_ENROLL_PHOTOS} face photos (got {n})",
        )

    with get_db() as conn:
        existing = get_student(conn, student_id)
        enrolled_date = today_str()
        if existing:
            conn.execute(
                """
                UPDATE students
                SET name = ?, class_section = ?, guardian_whatsapp_number = ?
                WHERE student_id = ?
                """,
                (
                    name,
                    section,
                    payload.guardian_whatsapp_number or None,
                    student_id,
                ),
            )
            conn.execute("DELETE FROM embeddings WHERE student_id = ?", (student_id,))
        else:
            conn.execute(
                """
                INSERT INTO students (student_id, name, class_section, guardian_whatsapp_number, enrolled_date)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    name,
                    section,
                    payload.guardian_whatsapp_number or None,
                    enrolled_date,
                ),
            )
        for i, item in enumerate(payload.embeddings, start=1):
            conn.execute(
                "INSERT INTO embeddings (student_id, embedding, created_at) VALUES (?, ?, ?)",
                (student_id, json.dumps(item.descriptor), now().isoformat(timespec="seconds")),
            )
            if item.photo_jpeg_base64:
                raw = item.photo_jpeg_base64.split(",", 1)[-1]
                try:
                    save_photo(student_id, i, base64.b64decode(raw))
                except Exception:
                    pass

    return {"ok": True, "student_id": student_id, "photos": n}


class RecognizeIn(BaseModel):
    descriptors: list[list[float]]
    liveness_ok: bool = True


@app.post("/api/recognize")
def api_recognize(payload: RecognizeIn):
    if not payload.liveness_ok:
        return JSONResponse(
            {"ok": False, "error": "liveness_failed", "matches": []},
            status_code=400,
        )
    if not payload.descriptors:
        return {"ok": True, "matches": [], "logged": []}

    with get_db() as conn:
        settings = get_settings(conn)
        gallery = load_all_embeddings(conn)
        threshold = float(settings.get("match_threshold", MATCH_THRESHOLD))
        matches = match_faces(
            payload.descriptors,
            gallery,
            cosine_threshold=threshold,
            euclidean_threshold=EUCLIDEAN_THRESHOLD,
        )
        logged = []
        for match in matches:
            if not match["matched"]:
                continue
            result = record_attendance(conn, match["student_id"], settings)
            match["attendance"] = result
            if result["recorded"]:
                logged.append(
                    {
                        "student_id": match["student_id"],
                        "name": match["name"],
                        "status": result["attendance"]["status"],
                        "time_in": result["attendance"]["time_in"],
                    }
                )
                add_scan_event(
                    conn,
                    kind="logged",
                    student_id=match["student_id"],
                    name=match["name"],
                    status=result["attendance"]["status"],
                    time_in=result["attendance"]["time_in"],
                    message=(
                        f"{match['name']} · {match['student_id']} marked "
                        f"{result['attendance']['status']} at "
                        f"{str(result['attendance']['time_in'] or '')[11:19]}"
                    ),
                )
            else:
                match["already_logged"] = True
    return {"ok": True, "matches": matches, "logged": logged}


@app.post("/api/close-day")
def api_close_day(day: str | None = Query(default=None)):
    day = day or today_str()
    with get_db() as conn:
        n = mark_absents_for_date(conn, day)
    return {"ok": True, "marked_absent": n, "date": day}


@app.delete("/api/students/{student_id}")
def api_delete_student(student_id: str):
    with get_db() as conn:
        cur = conn.execute("DELETE FROM students WHERE student_id = ?", (student_id.upper(),))
        if cur.rowcount == 0:
            raise HTTPException(404, "Student not found")
    folder = PHOTO_DIR / student_id.upper()
    if folder.exists():
        for f in folder.iterdir():
            f.unlink()
        folder.rmdir()
    return {"ok": True}


@app.get("/api/snapshot")
def api_snapshot():
    with get_db() as conn:
        snap = export_snapshot(conn)
        snap["meta"] = {
            "student_count": len(snap["students"]),
            "attendance_count": len(snap["attendance"]),
            "event_count": len(snap["events"]),
        }
        return snap


@app.get("/api/events")
def api_events(limit: int = Query(default=80, ge=1, le=300)):
    with get_db() as conn:
        return {"ok": True, "events": list_scan_events(conn, limit=limit)}


class RestoreIn(BaseModel):
    students: list[dict[str, Any]] = []
    attendance: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []


@app.post("/api/restore")
def api_restore(payload: RestoreIn):
    with get_db() as conn:
        counts = restore_snapshot(conn, payload.model_dump())
        snap = export_snapshot(conn)
    return {
        "ok": True,
        "restored": counts,
        "meta": {
            "student_count": len(snap["students"]),
            "attendance_count": len(snap["attendance"]),
        },
    }


@app.get("/export/daily.csv")
def export_daily_csv(day: str | None = None, class_section: str | None = None):
    day = day or today_str()
    with get_db() as conn:
        rows = daily_report(conn, day, class_section)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["student_id", "name", "class_section", "date", "time_in", "status"])
    for r in rows:
        writer.writerow(
            [
                r["student_id"],
                r["name"],
                r["class_section"],
                day,
                r["time_in"] or "",
                r["status"],
            ]
        )
    buf.seek(0)
    filename = f"attendance-{day}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/daily.pdf")
def export_daily_pdf(day: str | None = None, class_section: str | None = None):
    day = day or today_str()
    with get_db() as conn:
        settings = get_settings(conn)
        rows = daily_report(conn, day, class_section)
    table = [
        [
            r["student_id"],
            r["name"],
            r["class_section"],
            (r["time_in"] or "—")[11:19] if r["time_in"] else "—",
            r["status"],
        ]
        for r in rows
    ]
    subtitle = f"{settings.get('institution', 'KIIT University')} · {day}"
    if class_section:
        subtitle += f" · {class_section}"
    pdf = attendance_pdf(
        "Daily attendance",
        subtitle,
        ["ID", "Name", "Class", "Time in", "Status"],
        table,
    )
    filename = f"attendance-{day}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/range.csv")
def export_range_csv(start: str, end: str, class_section: str | None = None):
    with get_db() as conn:
        rows = range_report(conn, start, end, class_section)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "student_id",
            "name",
            "class_section",
            "present_days",
            "late_days",
            "absent_days",
            "span_days",
            "attendance_pct",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r["student_id"],
                r["name"],
                r["class_section"],
                r["present_days"] or 0,
                r["late_days"] or 0,
                r["absent_days"] or 0,
                r["span_days"],
                r["attendance_pct"],
            ]
        )
    buf.seek(0)
    filename = f"attendance-{start}-to-{end}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/range.pdf")
def export_range_pdf(start: str, end: str, class_section: str | None = None):
    with get_db() as conn:
        settings = get_settings(conn)
        rows = range_report(conn, start, end, class_section)
    table = [
        [
            r["student_id"],
            r["name"],
            r["class_section"],
            str(r["present_days"] or 0),
            str(r["late_days"] or 0),
            str(r["absent_days"] or 0),
            f"{r['attendance_pct']}%",
        ]
        for r in rows
    ]
    subtitle = f"{settings.get('institution', 'KIIT University')} · {start} to {end}"
    if class_section:
        subtitle += f" · {class_section}"
    pdf = attendance_pdf(
        "Attendance summary",
        subtitle,
        ["ID", "Name", "Class", "Present", "Late", "Absent", "%"],
        table,
    )
    filename = f"attendance-{start}-to-{end}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/health")
def health():
    return {"ok": True}
