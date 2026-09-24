"""The shipped bot must not regress against its committed baseline."""
import json
from pathlib import Path

from evalharness.cli import run
from evalharness.regression import compare

ROOT = Path(__file__).resolve().parents[1]


def test_guarded_bot_meets_baseline(tmp_path):
    summary = run("guarded", ROOT / "data/golden_set.jsonl", ROOT / "data/knowledge_base", tmp_path)
    baseline = json.loads((ROOT / "baselines/guarded.json").read_text())
    assert compare(summary, baseline) == []
