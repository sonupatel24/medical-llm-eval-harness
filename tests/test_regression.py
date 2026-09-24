from evalharness.regression import compare


def test_no_regression_within_tolerance():
    assert compare({"correctness": 0.90}, {"correctness": 0.91}) == []


def test_drop_is_flagged():
    assert compare({"correctness": 0.80}, {"correctness": 0.90})


def test_hallucination_rise_is_flagged():
    assert compare({"hallucination_rate": 0.10}, {"hallucination_rate": 0.0})
