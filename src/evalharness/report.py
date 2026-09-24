"""Markdown report for a run."""
from __future__ import annotations


def render(bot: str, summary: dict, results: list[dict]) -> str:
    lines = [f"# Evaluation report - `{bot}` bot", "", "| Metric | Value |", "|---|---|"]
    for k, v in summary.items():
        if k != "by_category":
            lines.append(f"| {k} | {v} |")
    lines += ["", "## Correctness by category", "", "| Category | Correctness |", "|---|---|"]
    lines += [f"| {k} | {v:.0%} |" for k, v in summary["by_category"].items()]
    failed = [r for r in results if not r["correct"]]
    lines += ["", f"## Failed cases ({len(failed)})", ""]
    if not failed:
        lines.append("None.")
    else:
        lines += ["| ID | Question | Answer |", "|---|---|---|"]
        lines += [f"| {r['id']} | {r['question']} | {r['answer']} |" for r in failed]
    return "\n".join(lines) + "\n"
