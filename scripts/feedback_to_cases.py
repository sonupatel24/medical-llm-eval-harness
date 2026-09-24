"""Turn thumbs-down user feedback into draft golden-set cases for review.

Input: JSONL feedback log, one object per line:
  {"question": "...", "answer": "...", "rating": "down", "comment": "..."}
Output: draft cases appended to data/candidate_cases.jsonl. A reviewer fills in
must_include / expect_refusal, then moves approved cases into golden_set.jsonl.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("feedback", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/candidate_cases.jsonl"))
    a = p.parse_args()
    seen = set()
    if a.out.exists():
        seen = {json.loads(l)["question"] for l in a.out.read_text().splitlines() if l.strip()}
    added = 0
    with a.out.open("a", encoding="utf-8") as f:
        for line in a.feedback.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fb = json.loads(line)
            if fb.get("rating") != "down" or fb["question"] in seen:
                continue
            seen.add(fb["question"])
            case = {
                "id": "fb_" + hashlib.sha1(fb["question"].encode()).hexdigest()[:8],
                "category": "from_feedback",
                "question": fb["question"],
                "must_include": [],
                "must_not_include": [],
                "expect_refusal": False,
                "expected_source": None,
                "bad_answer": fb.get("answer"),
                "reviewer_note": fb.get("comment", ""),
            }
            f.write(json.dumps(case) + "\n")
            added += 1
    print(f"Added {added} draft case(s) to {a.out}")


if __name__ == "__main__":
    main()
