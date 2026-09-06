from __future__ import annotations

import sqlite3
from datetime import datetime

from app.db import TZ, now, today_str


def parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":")
    return int(hour), int(minute)


def status_for_time(detected: datetime, settings: dict[str, str]) -> str:
    """Present inside the attendance window; Late after it. Always keep real time_in."""
    late_h, late_m = parse_hhmm(settings.get("late_after", "08:30"))
    late_at = detected.replace(hour=late_h, minute=late_m, second=0, microsecond=0)
    if detected <= late_at:
        return "Present"
    return "Late"


def record_attendance(
    conn: sqlite3.Connection,
    student_id: str,
    settings: dict[str, str],
    detected_at: datetime | None = None,
) -> dict:
    detected = detected_at or now()
    day = detected.astimezone(TZ).date().isoformat()
    existing = conn.execute(
        "SELECT * FROM attendance WHERE student_id = ? AND date = ?",
        (student_id, day),
    ).fetchone()
    if existing:
        return {
            "recorded": False,
            "reason": "already_logged",
            "attendance": dict(existing),
        }

    status = status_for_time(detected, settings)
    time_in = detected.isoformat(timespec="seconds")
    try:
        conn.execute(
            """
            INSERT INTO attendance (student_id, date, time_in, status)
            VALUES (?, ?, ?, ?)
            """,
            (student_id, day, time_in, status),
        )
    except sqlite3.IntegrityError:
        row = conn.execute(
            "SELECT * FROM attendance WHERE student_id = ? AND date = ?",
            (student_id, day),
        ).fetchone()
        return {
            "recorded": False,
            "reason": "already_logged",
            "attendance": dict(row) if row else None,
        }

    row = conn.execute(
        "SELECT * FROM attendance WHERE student_id = ? AND date = ?",
        (student_id, day),
    ).fetchone()
    return {"recorded": True, "reason": "inserted", "attendance": dict(row)}


def mark_absents_for_date(conn: sqlite3.Connection, day: str | None = None) -> int:
    day = day or today_str()
    missing = conn.execute(
        """
        SELECT s.student_id
        FROM students s
        LEFT JOIN attendance a ON a.student_id = s.student_id AND a.date = ?
        WHERE a.attendance_id IS NULL
        """,
        (day,),
    ).fetchall()
    count = 0
    for row in missing:
        conn.execute(
            """
            INSERT OR IGNORE INTO attendance (student_id, date, time_in, status)
            VALUES (?, ?, NULL, 'Absent')
            """,
            (row["student_id"], day),
        )
        count += 1
    return count


def daily_report(
    conn: sqlite3.Connection, day: str, class_section: str | None = None
) -> list[sqlite3.Row]:
    params: list[str] = [day]
    where = ""
    if class_section:
        where = "AND s.class_section = ?"
        params.append(class_section)
    return conn.execute(
        f"""
        SELECT
            s.student_id,
            s.name,
            s.class_section,
            s.guardian_whatsapp_number,
            a.attendance_id,
            a.date,
            a.time_in,
            COALESCE(a.status, 'Absent') AS status
        FROM students s
        LEFT JOIN attendance a ON a.student_id = s.student_id AND a.date = ?
        WHERE 1=1 {where}
        ORDER BY s.class_section, s.name
        """,
        params,
    ).fetchall()


def range_report(
    conn: sqlite3.Connection,
    start: str,
    end: str,
    class_section: str | None = None,
) -> list[dict]:
    params: list[str] = [start, end]
    where = ""
    if class_section:
        where = "AND s.class_section = ?"
        params.append(class_section)
    rows = conn.execute(
        f"""
        SELECT
            s.student_id,
            s.name,
            s.class_section,
            COUNT(a.attendance_id) AS logged_days,
            SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) AS present_days,
            SUM(CASE WHEN a.status = 'Late' THEN 1 ELSE 0 END) AS late_days,
            SUM(CASE WHEN a.status = 'Absent' THEN 1 ELSE 0 END) AS absent_days
        FROM students s
        LEFT JOIN attendance a
            ON a.student_id = s.student_id AND a.date >= ? AND a.date <= ?
        WHERE 1=1 {where}
        GROUP BY s.student_id
        ORDER BY s.class_section, s.name
        """,
        params,
    ).fetchall()

    from datetime import date as date_cls

    start_d = date_cls.fromisoformat(start)
    end_d = date_cls.fromisoformat(end)
    span = (end_d - start_d).days + 1
    span = max(span, 1)

    out = []
    for row in rows:
        present_like = (row["present_days"] or 0) + (row["late_days"] or 0)
        pct = round(100.0 * present_like / span, 1)
        out.append(
            {
                **dict(row),
                "span_days": span,
                "attendance_pct": pct,
            }
        )
    out.sort(key=lambda r: (r["attendance_pct"], r["name"]))
    return out
