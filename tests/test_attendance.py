from app.attendance import record_attendance, status_for_time
from app.db import connect, get_settings, init_db, now
from app.recognition import cosine_similarity, match_faces


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("app.db.DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr("app.db.DATA_DIR", tmp_path)
    monkeypatch.setattr("app.db.PHOTO_DIR", tmp_path / "photos")
    init_db()
    conn = connect()
    conn.execute(
        "INSERT INTO students (student_id, name, class_section, enrolled_date) VALUES (?,?,?,?)",
        ("CSE001", "Asha Rao", "CSE-4A", "2026-09-01"),
    )
    conn.commit()
    return conn


def test_one_attendance_per_day(tmp_path, monkeypatch):
    conn = setup(tmp_path, monkeypatch)
    settings = get_settings(conn)
    first = record_attendance(conn, "CSE001", settings)
    conn.commit()
    assert first["recorded"] is True
    second = record_attendance(conn, "CSE001", settings)
    conn.commit()
    assert second["recorded"] is False
    assert second["reason"] == "already_logged"
    n = conn.execute("SELECT COUNT(*) AS n FROM attendance").fetchone()["n"]
    assert n == 1
    conn.close()


def test_unique_constraint(tmp_path, monkeypatch):
    conn = setup(tmp_path, monkeypatch)
    settings = get_settings(conn)
    record_attendance(conn, "CSE001", settings)
    conn.commit()
    import sqlite3

    try:
        conn.execute(
            "INSERT INTO attendance (student_id, date, time_in, status) VALUES (?,?,?,?)",
            ("CSE001", now().date().isoformat(), now().isoformat(), "Present"),
        )
        conn.commit()
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    assert raised
    conn.close()


def test_late_status():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Asia/Kolkata")
    settings = {"late_after": "08:30"}
    early = datetime(2026, 9, 6, 8, 12, tzinfo=tz)
    late = datetime(2026, 9, 6, 9, 5, tzinfo=tz)
    assert status_for_time(early, settings) == "Present"
    assert status_for_time(late, settings) == "Late"


def test_match_multiple_faces():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    gallery = [
        {"student_id": "A", "name": "Ann", "class_section": "1", "embedding": a},
        {"student_id": "B", "name": "Ben", "class_section": "1", "embedding": b},
    ]
    matches = match_faces([a, b], gallery, 0.5, 0.2)
    ids = {m["student_id"] for m in matches if m["matched"]}
    assert ids == {"A", "B"}


def test_cosine():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
