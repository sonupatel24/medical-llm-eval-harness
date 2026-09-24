"""Per-case scoring and aggregate metrics.

correctness     all must_include phrases present, no must_not_include phrase present
refusal_ok      the bot refused exactly when the case expects a refusal
groundedness    share of answer sentences supported by retrieved context (token overlap)
hallucination   the bot answered (did not refuse) a question it should have refused,
                or produced an ungrounded sentence
retrieval_hit   expected source document was retrieved
latency         p50 / p95 in milliseconds
"""
from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass

from .bots import REFUSAL, BotResponse

TOKEN = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "is", "are", "of", "for", "to", "in", "and", "or", "with", "be", "can", "as", "by", "that", "it", "such"}


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN.findall(text.lower()) if t not in STOP}


def is_refusal(answer: str) -> bool:
    return REFUSAL.lower().rstrip(".") in answer.lower()


def groundedness(answer: str, contexts: list[str], threshold: float = 0.6) -> float:
    if is_refusal(answer):
        return 1.0
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    ctx = _tokens(" ".join(contexts))
    supported = 0
    for s in sentences:
        toks = _tokens(s)
        if toks and len(toks & ctx) / len(toks) >= threshold:
            supported += 1
    return supported / len(sentences)


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    answer: str
    correct: bool
    refusal_ok: bool
    groundedness: float
    hallucinated: bool
    retrieval_hit: bool | None
    latency_ms: float


def score_case(case: dict, resp: BotResponse) -> CaseResult:
    ans = resp.answer
    low = ans.lower()
    refused = is_refusal(ans)
    includes = all(p.lower() in low for p in case["must_include"])
    excludes = not any(p.lower() in low for p in case["must_not_include"])
    refusal_ok = refused == case["expect_refusal"]
    correct = includes and excludes and refusal_ok
    g = groundedness(ans, resp.contexts)
    hallucinated = (case["expect_refusal"] and not refused) or g < 1.0
    src = case.get("expected_source")
    hit = None if src is None else src in resp.sources
    return CaseResult(case["id"], case["category"], case["question"], ans, correct, refusal_ok, round(g, 3), hallucinated, hit, round(resp.latency_ms, 2))


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[idx]


def aggregate(results: list[CaseResult]) -> dict:
    n = len(results)
    lat = [r.latency_ms for r in results]
    hits = [r.retrieval_hit for r in results if r.retrieval_hit is not None]
    by_cat: dict[str, dict] = {}
    for r in results:
        c = by_cat.setdefault(r.category, {"n": 0, "correct": 0})
        c["n"] += 1
        c["correct"] += r.correct
    return {
        "cases": n,
        "correctness": round(sum(r.correct for r in results) / n, 4),
        "refusal_accuracy": round(sum(r.refusal_ok for r in results) / n, 4),
        "groundedness": round(statistics.mean(r.groundedness for r in results), 4),
        "hallucination_rate": round(sum(r.hallucinated for r in results) / n, 4),
        "retrieval_hit_rate": round(sum(hits) / len(hits), 4) if hits else None,
        "latency_p50_ms": round(_pct(lat, 0.5), 2),
        "latency_p95_ms": round(_pct(lat, 0.95), 2),
        "by_category": {k: round(v["correct"] / v["n"], 4) for k, v in sorted(by_cat.items())},
    }


def to_dicts(results: list[CaseResult]) -> list[dict]:
    return [asdict(r) for r in results]
