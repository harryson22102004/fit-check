"""WER / BLEU-style evaluation helpers (no extra metric deps required)."""

from __future__ import annotations

import re


def tokens(s: str) -> list[str]:
    return re.findall(r"\w+", (s or "").lower(), flags=re.UNICODE)


def wer(ref: str, hyp: str) -> float:
    r, h = tokens(ref), tokens(hyp)
    if not r:
        return 0.0 if not h else 1.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / len(r)


def bleu1(ref: str, hyp: str) -> float:
    r, h = tokens(ref), tokens(hyp)
    if not h:
        return 0.0
    from collections import Counter

    rc, hc = Counter(r), Counter(h)
    overlap = sum(min(hc[w], rc[w]) for w in hc)
    return overlap / max(len(h), 1)
