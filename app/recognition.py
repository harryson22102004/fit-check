from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def euclidean_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return float("inf")
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def match_faces(
    detections: list[list[float]],
    gallery: list[dict[str, Any]],
    cosine_threshold: float,
    euclidean_threshold: float,
) -> list[dict[str, Any]]:
    """Match each detected embedding to the best enrolled student.

    Multiple faces in one frame are handled independently. A student can
    match at most once per frame (highest score wins if two detections
    collide on the same person).
    """
    claimed: set[str] = set()
    results: list[dict[str, Any]] = []

    by_student: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in gallery:
        by_student[item["student_id"]].append(item)

    scored: list[tuple[float, int, str, dict[str, Any]]] = []
    for idx, vec in enumerate(detections):
        best_id = None
        best_score = -1.0
        best_meta: dict[str, Any] | None = None
        best_euc = float("inf")
        for student_id, items in by_student.items():
            for item in items:
                score = cosine_similarity(vec, item["embedding"])
                dist = euclidean_distance(vec, item["embedding"])
                if score > best_score:
                    best_score = score
                    best_euc = dist
                    best_id = student_id
                    best_meta = item
        if best_id and best_meta and (
            best_score >= cosine_threshold or best_euc <= euclidean_threshold
        ):
            scored.append((best_score, idx, best_id, best_meta))
        else:
            results.append(
                {
                    "face_index": idx,
                    "matched": False,
                    "student_id": None,
                    "name": None,
                    "class_section": None,
                    "similarity": round(best_score, 4) if best_score >= 0 else 0.0,
                    "distance": round(best_euc, 4) if best_euc != float("inf") else None,
                }
            )

    scored.sort(reverse=True)
    matched_indices: set[int] = set()
    for score, idx, student_id, meta in scored:
        if student_id in claimed:
            results.append(
                {
                    "face_index": idx,
                    "matched": False,
                    "student_id": None,
                    "name": None,
                    "class_section": None,
                    "similarity": round(score, 4),
                    "distance": None,
                    "reason": "duplicate_in_frame",
                }
            )
            continue
        claimed.add(student_id)
        matched_indices.add(idx)
        results.append(
            {
                "face_index": idx,
                "matched": True,
                "student_id": student_id,
                "name": meta["name"],
                "class_section": meta["class_section"],
                "similarity": round(score, 4),
            }
        )

    results.sort(key=lambda r: r["face_index"])
    return results
