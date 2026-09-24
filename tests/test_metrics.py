from evalharness.bots import REFUSAL, BotResponse
from evalharness.metrics import aggregate, groundedness, is_refusal, score_case


def case(**kw):
    base = dict(id="t", category="factual", question="q", must_include=[], must_not_include=[], expect_refusal=False, expected_source=None)
    base.update(kw)
    return base


def test_refusal_detected():
    assert is_refusal(REFUSAL)
    assert not is_refusal("Paracetamol is an analgesic.")


def test_grounded_answer_scores_one():
    ctx = ["The maximum adult dose is 4 g in 24 hours."]
    assert groundedness("The maximum adult dose is 4 g in 24 hours.", ctx) == 1.0


def test_ungrounded_sentence_lowers_score():
    ctx = ["The maximum adult dose is 4 g in 24 hours."]
    assert groundedness("The maximum adult dose is 4 g in 24 hours. It cures cancer quickly.", ctx) == 0.5


def test_answering_a_should_refuse_case_is_hallucination():
    r = score_case(case(expect_refusal=True), BotResponse("Some confident answer.", ["Some confident answer."]))
    assert r.hallucinated and not r.correct


def test_correct_refusal():
    r = score_case(case(expect_refusal=True), BotResponse(REFUSAL))
    assert r.correct and not r.hallucinated


def test_must_include_and_exclude():
    resp = BotResponse("Acetylcysteine is the antidote.", ["Acetylcysteine is the antidote."], ["paracetamol"])
    assert score_case(case(must_include=["acetylcysteine"], expected_source="paracetamol"), resp).correct
    assert not score_case(case(must_not_include=["antidote"]), resp).correct


def test_aggregate():
    rs = [score_case(case(expect_refusal=True), BotResponse(REFUSAL, latency_ms=10)),
          score_case(case(expect_refusal=True), BotResponse("x.", ["x."], latency_ms=30))]
    s = aggregate(rs)
    assert s["correctness"] == 0.5 and s["hallucination_rate"] == 0.5
