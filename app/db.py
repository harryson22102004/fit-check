from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from app.config import DATA_DIR, DB_PATH, DEFAULT_SETTINGS, PHOTO_DIR, TIMEZONE

TZ = ZoneInfo(TIMEZONE)


def now() -> datetime:
    return datetime.now(TZ)


def today_str() -> str:
    return now().date().isoformat()


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                class_section TEXT NOT NULL,
                guardian_whatsapp_number TEXT,
                enrolled_date TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                date TEXT NOT NULL,
                time_in TEXT,
                status TEXT NOT NULL CHECK (status IN ('Present', 'Late', 'Absent')),
                UNIQUE (student_id, date)
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scan_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                name TEXT,
                kind TEXT NOT NULL,
                status TEXT,
                time_in TEXT,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
            CREATE INDEX IF NOT EXISTS idx_embeddings_student ON embeddings(student_id);
            CREATE INDEX IF NOT EXISTS idx_scan_events_created ON scan_events(created_at);
            """
        )
        for key, value in DEFAULT_SETTINGS.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )


def get_settings(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {row["key"]: row["value"] for row in rows}


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def list_students(conn: sqlite3.Connection, class_section: str | None = None) -> list[sqlite3.Row]:
    if class_section:
        return conn.execute(
            """
            SELECT s.*, COUNT(e.id) AS embedding_count
            FROM students s
            LEFT JOIN embeddings e ON e.student_id = s.student_id
            WHERE s.class_section = ?
            GROUP BY s.student_id
            ORDER BY s.class_section, s.name
            """,
            (class_section,),
        ).fetchall()
    return conn.execute(
        """
        SELECT s.*, COUNT(e.id) AS embedding_count
        FROM students s
        LEFT JOIN embeddings e ON e.student_id = s.student_id
        GROUP BY s.student_id
        ORDER BY s.class_section, s.name
        """
    ).fetchall()


def get_student(conn: sqlite3.Connection, student_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT s.*, COUNT(e.id) AS embedding_count
        FROM students s
        LEFT JOIN embeddings e ON e.student_id = s.student_id
        WHERE s.student_id = ?
        GROUP BY s.student_id
        """,
        (student_id,),
    ).fetchone()


def class_sections(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT class_section FROM students ORDER BY class_section"
    ).fetchall()
    return [row["class_section"] for row in rows]


def student_photo_paths(student_id: str) -> list[str]:
    folder = PHOTO_DIR / student_id
    if not folder.exists():
        return []
    files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    return [f"/photos/{student_id}/{p.name}" for p in files]


def save_photo(student_id: str, index: int, data: bytes) -> Path:
    folder = PHOTO_DIR / student_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{index:02d}.jpg"
    path.write_bytes(data)
    return path


def load_all_embeddings(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT e.student_id, e.embedding, s.name, s.class_section
        FROM embeddings e
        JOIN students s ON s.student_id = e.student_id
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "student_id": row["student_id"],
                "name": row["name"],
                "class_section": row["class_section"],
                "embedding": json.loads(row["embedding"]),
            }
        )
    return out


def add_scan_event(
    conn: sqlite3.Connection,
    *,
    kind: str,
    message: str,
    student_id: str | None = None,
    name: str | None = None,
    status: str | None = None,
    time_in: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO scan_events (student_id, name, kind, status, time_in, message, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            student_id,
            name,
            kind,
            status,
            time_in,
            message,
            now().isoformat(timespec="seconds"),
        ),
    )


def list_scan_events(conn: sqlite3.Connection, limit: int = 80) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT student_id, name, kind, status, time_in, message, created_at
        FROM scan_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def student_attendance_history(conn: sqlite3.Connection, student_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT date, time_in, status
        FROM attendance
        WHERE student_id = ?
        ORDER BY date DESC
        """,
        (student_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def export_snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    students = []
    for row in conn.execute(
        """
        SELECT student_id, name, class_section, guardian_whatsapp_number, enrolled_date
        FROM students
        ORDER BY name
        """
    ):
        embeddings = [
            json.loads(e["embedding"])
            for e in conn.execute(
                "SELECT embedding FROM embeddings WHERE student_id = ? ORDER BY id",
                (row["student_id"],),
            )
        ]
        students.append(
            {
                "student_id": row["student_id"],
                "name": row["name"],
                "class_section": row["class_section"],
                "guardian_whatsapp_number": row["guardian_whatsapp_number"],
                "enrolled_date": row["enrolled_date"],
                "embeddings": embeddings,
            }
        )
    attendance = [
        dict(r)
        for r in conn.execute(
            "SELECT student_id, date, time_in, status FROM attendance ORDER BY date, student_id"
        )
    ]
    events = list_scan_events(conn, limit=200)
    return {
        "students": students,
        "attendance": attendance,
        "events": events,
    }


def restore_snapshot(conn: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, int]:
    students_n = 0
    attendance_n = 0
    events_n = 0
    for s in payload.get("students") or []:
        sid = str(s.get("student_id", "")).strip().upper()
        name = str(s.get("name", "")).strip()
        section = str(s.get("class_section", "")).strip()
        if not sid or not name or not section:
            continue
        enrolled = s.get("enrolled_date") or today_str()
        conn.execute(
            """
            INSERT INTO students (student_id, name, class_section, guardian_whatsapp_number, enrolled_date)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                name = excluded.name,
                class_section = excluded.class_section,
                guardian_whatsapp_number = excluded.guardian_whatsapp_number
            """,
            (sid, name, section, s.get("guardian_whatsapp_number"), enrolled),
        )
        embeddings = s.get("embeddings") or []
        if embeddings:
            conn.execute("DELETE FROM embeddings WHERE student_id = ?", (sid,))
            for vec in embeddings:
                if not isinstance(vec, list) or len(vec) < 64:
                    continue
                conn.execute(
                    "INSERT INTO embeddings (student_id, embedding, created_at) VALUES (?, ?, ?)",
                    (sid, json.dumps(vec), now().isoformat(timespec="seconds")),
                )
        students_n += 1

    for row in payload.get("attendance") or []:
        sid = str(row.get("student_id", "")).strip().upper()
        day = row.get("date")
        status = row.get("status")
        if not sid or not day or status not in ("Present", "Late", "Absent"):
            continue
        exists = conn.execute(
            "SELECT 1 FROM students WHERE student_id = ?", (sid,)
        ).fetchone()
        if not exists:
            continue
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO attendance (student_id, date, time_in, status)
            VALUES (?, ?, ?, ?)
            """,
            (sid, day, row.get("time_in"), status),
        )
        attendance_n += cur.rowcount

    existing_events = conn.execute("SELECT COUNT(*) AS n FROM scan_events").fetchone()["n"]
    if existing_events == 0:
        for ev in reversed(payload.get("events") or []):
            msg = ev.get("message")
            if not msg:
                continue
            conn.execute(
                """
                INSERT INTO scan_events (student_id, name, kind, status, time_in, message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ev.get("student_id"),
                    ev.get("name"),
                    ev.get("kind") or "note",
                    ev.get("status"),
                    ev.get("time_in"),
                    msg,
                    ev.get("created_at") or now().isoformat(timespec="seconds"),
                ),
            )
            events_n += 1
    return {"students": students_n, "attendance": attendance_n, "events": events_n}

