"""Command line: run an evaluation, compare to baseline, or save a new baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bots import build_bot
from .metrics import aggregate, score_case, to_dicts
from .regression import compare
from .report import render
from .retriever import Retriever

ROOT = Path(__file__).resolve().parents[2]


def load_cases(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run(bot_name: str, cases_path: Path, kb: Path, out_dir: Path) -> dict:
    bot = build_bot(bot_name, Retriever(kb))
    results = [score_case(c, bot.ask(c["question"])) for c in load_cases(cases_path)]
    summary = aggregate(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = to_dicts(results)
    (out_dir / f"{bot_name}_results.json").write_text(json.dumps({"summary": summary, "results": rows}, indent=2))
    (out_dir / f"{bot_name}_report.md").write_text(render(bot_name, summary, rows))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evalharness")
    p.add_argument("command", choices=["run", "check", "save-baseline"])
    p.add_argument("--bot", default="extractive", choices=["extractive", "guarded", "llm"])
    p.add_argument("--cases", type=Path, default=ROOT / "data/golden_set.jsonl")
    p.add_argument("--kb", type=Path, default=ROOT / "data/knowledge_base")
    p.add_argument("--out", type=Path, default=ROOT / "reports")
    p.add_argument("--baseline", type=Path, default=None)
    a = p.parse_args(argv)
    baseline = a.baseline or ROOT / f"baselines/{a.bot}.json"

    summary = run(a.bot, a.cases, a.kb, a.out)
    print(json.dumps(summary, indent=2))

    if a.command == "save-baseline":
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text(json.dumps(summary, indent=2))
        print(f"Saved baseline to {baseline}")
    elif a.command == "check":
        failures = compare(summary, json.loads(baseline.read_text()))
        if failures:
            print("REGRESSION DETECTED:\n  " + "\n  ".join(failures))
            return 1
        print("No regressions against baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
