from fastapi.testclient import TestClient

from app.db import init_db
from app.main import app


def test_enroll_and_recognize_once_per_day(tmp_path, monkeypatch):
    monkeypatch.setattr("app.db.DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr("app.db.DATA_DIR", tmp_path)
    monkeypatch.setattr("app.db.PHOTO_DIR", tmp_path / "photos")
    monkeypatch.setattr("app.main.PHOTO_DIR", tmp_path / "photos")
    (tmp_path / "photos").mkdir()
    init_db()

    vec = [0.0] * 127 + [1.0]
    other = [1.0] + [0.0] * 127
    with TestClient(app) as client:
        res = client.post(
            "/api/enroll",
            json={
                "student_id": "KIIT2401",
                "name": "Vivek Das",
                "class_section": "CSE-4A",
                "embeddings": [{"descriptor": vec} for _ in range(5)],
            },
        )
        assert res.status_code == 200, res.text

        first = client.post("/api/recognize", json={"descriptors": [vec], "liveness_ok": True})
        assert first.status_code == 200
        body = first.json()
        assert body["logged"][0]["student_id"] == "KIIT2401"
        assert body["logged"][0]["status"] in ("Present", "Late")

        second = client.post("/api/recognize", json={"descriptors": [vec], "liveness_ok": True})
        assert second.json()["logged"] == []
        assert second.json()["matches"][0]["already_logged"] is True

        unknown = client.post("/api/recognize", json={"descriptors": [other], "liveness_ok": True})
        assert unknown.json()["matches"][0]["matched"] is False

        roll = client.get("/")
        assert roll.status_code == 200
        assert "Vivek Das" in roll.text

        snap = client.get("/api/snapshot").json()
        assert snap["meta"]["student_count"] == 1
        assert snap["meta"]["attendance_count"] == 1
        assert snap["events"]

        empty = tmp_path / "empty.db"
        monkeypatch.setattr("app.db.DB_PATH", empty)
        init_db()
        restored = client.post("/api/restore", json=snap)
        assert restored.status_code == 200
        hist = client.get("/history")
        assert hist.status_code == 200
        assert "Vivek Das" in hist.text
        person = client.get("/students/KIIT2401")
        assert person.status_code == 200
        assert "Present" in person.text or "Late" in person.text
