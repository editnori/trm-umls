#!/usr/bin/env python3
"""
Score a manually-labeled review set against an eval JSONL.

Review set format (JSONL)
  - Produced by `make_review_set.py`
  - Each row should have:
      - id
      - note_file
      - extraction.{start,end,cui}
      - label.gold_cui (filled by human, ideally from candidates)

Scoring
  - Supports `label.gold_cui = NONE` to mark rows that should NOT be extracted.

Metrics
  - overall_accuracy: counts NONE rows (no-extract expected)
  - positive_accuracy: only rows with a real gold CUI (linking quality)
  - false_positives: predicted a CUI when gold is NONE
  - false_negatives: missing prediction when gold is a real CUI
  - wrong_link: predicted CUI != gold (both present)
  - gold_in_topk: whether gold appears anywhere in candidates list (when available)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _key(row: Dict[str, Any]) -> Tuple[str, int, int]:
    ex = row.get("extraction") or {}
    return str(row.get("note_file")), int(ex.get("start", -1)), int(ex.get("end", -1))


def _gold_cui(row: Dict[str, Any]) -> str:
    lab = row.get("label") or {}
    return str(lab.get("gold_cui") or "").strip()


def _is_none_gold(gold: str) -> bool:
    g = (gold or "").strip().upper()
    return g in {"NONE", "NO", "NULL", "N/A"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a labeled review set against an eval JSONL")
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True, help="Eval output to score (JSONL)")
    args = parser.parse_args()

    review_rows = list(_load_jsonl(args.review_jsonl))
    if not review_rows:
        raise SystemExit(f"No rows in {args.review_jsonl}")

    eval_rows = list(_load_jsonl(args.eval_jsonl))
    if not eval_rows:
        raise SystemExit(f"No rows in {args.eval_jsonl}")

    eval_map: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for r in eval_rows:
        eval_map[_key(r)] = r

    total = 0
    labeled = 0
    labeled_positive = 0
    labeled_none = 0

    correct = 0
    correct_positive = 0
    false_positives = 0
    false_negatives = 0
    wrong_link = 0
    gold_in_topk = 0

    by_type = Counter()
    by_type_correct = Counter()

    for r in review_rows:
        total += 1
        gold = _gold_cui(r)
        if not gold:
            continue
        labeled += 1
        if _is_none_gold(gold):
            labeled_none += 1
        else:
            labeled_positive += 1

        k = _key(r)
        pred_row = eval_map.get(k)
        pred_cui = str(((pred_row or {}).get("extraction") or {}).get("cui") or "")

        note_type = str((pred_row or {}).get("note_type") or r.get("note_type") or "Unknown")
        by_type[note_type] += 1

        if _is_none_gold(gold):
            # Gold expects no extraction.
            if not pred_cui:
                correct += 1
                by_type_correct[note_type] += 1
            else:
                false_positives += 1
            continue

        # Gold expects a real CUI.
        if not pred_cui:
            false_negatives += 1
            continue

        if pred_cui == gold:
            correct += 1
            correct_positive += 1
            by_type_correct[note_type] += 1
        else:
            wrong_link += 1

        cands = (((pred_row or {}).get("extraction") or {}).get("candidates") or [])
        if any(str(c.get("cui") or "") == gold for c in cands):
            gold_in_topk += 1

    print(f"review_rows={total}")
    print(f"labeled_rows={labeled}")
    print(f"  labeled_positive={labeled_positive}")
    print(f"  labeled_none={labeled_none}")
    if labeled > 0:
        print(f"overall_accuracy={correct}/{labeled} ({correct/labeled:.3f})")
    if labeled_positive > 0:
        print(f"positive_accuracy={correct_positive}/{labeled_positive} ({correct_positive/labeled_positive:.3f})")
        print(f"gold_in_topk={gold_in_topk}/{labeled_positive} ({gold_in_topk/labeled_positive:.3f})")
    print(f"false_positives={false_positives}")
    print(f"false_negatives={false_negatives}")
    print(f"wrong_link={wrong_link}")

    if by_type:
        print("\nby_note_type:")
        for t, n in sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0])):
            c = int(by_type_correct.get(t, 0))
            acc = c / n if n else 0.0
            print(f"  {t}: {c}/{n} ({acc:.3f})")


if __name__ == "__main__":
    main()
