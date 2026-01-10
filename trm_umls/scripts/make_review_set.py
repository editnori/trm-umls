#!/usr/bin/env python3
"""
Create a small, human-reviewable concept linking set from an eval JSONL.

Input
  - An eval JSONL produced by `trm_umls/scripts/eval_notes_dir.py` with:
      - `context.{left,span,right}`
      - `extraction.candidates` (when `--include-candidates` was used)

Output
  - A JSONL with the same rows plus a `label` object the human can fill:
      - label.gold_cui: pick from candidates (recommended)
      - label.notes: free text

This is meant to be edited by hand and then scored with `score_review_set.py`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _key(row: Dict[str, Any]) -> Tuple[str, int, int]:
    ex = row.get("extraction") or {}
    return str(row.get("note_file")), int(ex.get("start", -1)), int(ex.get("end", -1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 50-row manual review set from an eval JSONL")
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rows = _load_jsonl(args.eval_jsonl)
    if not rows:
        raise SystemExit(f"No rows found in {args.eval_jsonl}")

    # Ensure we have candidates; otherwise the reviewer can't pick a gold concept quickly.
    missing_cands = sum(1 for r in rows if not (r.get("extraction") or {}).get("candidates"))
    if missing_cands:
        raise SystemExit(
            f"{missing_cands}/{len(rows)} rows are missing extraction.candidates. "
            "Re-run eval with --include-candidates."
        )

    # De-dupe by (note_file,start,end).
    uniq: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for r in rows:
        uniq[_key(r)] = r
    rows = list(uniq.values())

    rng = random.Random(int(args.seed))

    weird_term_re = re.compile(
        r"(?:\\bctcae\\b|\\brndx\\b|\\breported\\s+by\\s+patient\\b|\\bhow\\s+often\\b|\\bquestion\\b|\\bwere\\s+you\\b|^when\\b)",
        flags=re.IGNORECASE,
    )

    def _is_weird_term(term: str) -> bool:
        return bool(weird_term_re.search(term or ""))

    def _rerank_scores(r: Dict[str, Any]) -> List[float]:
        ex = r.get("extraction") or {}
        cands = ex.get("candidates") or []
        out: List[float] = []
        for c in cands:
            try:
                out.append(float(c.get("rerank_score", c.get("score", 0.0))))
            except Exception:
                out.append(0.0)
        return out

    def _rerank_margin(r: Dict[str, Any]) -> float:
        scores = sorted(_rerank_scores(r), reverse=True)
        if len(scores) < 2:
            return 999.0
        return float(scores[0] - scores[1])

    def _pred_candidate(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ex = r.get("extraction") or {}
        pred_cui = str(ex.get("cui") or "")
        for c in (ex.get("candidates") or []):
            if str(c.get("cui") or "") == pred_cui:
                return c
        return None

    def _tricky_score(r: Dict[str, Any]) -> float:
        ex = r.get("extraction") or {}
        pred_term = str(ex.get("preferred_term") or "")
        pred_group = str(ex.get("semantic_group") or "")
        pred_tui = str(ex.get("tui") or "")
        base_score = float(ex.get("score") or 0.0)
        cands = ex.get("candidates") or []

        score = 0.0
        if _is_weird_term(pred_term):
            score += 10.0
        if any(_is_weird_term(str(c.get("preferred_term") or "")) for c in cands[: min(15, len(cands))]):
            score += 5.0

        margin = _rerank_margin(r)
        if margin < 0.010:
            score += 5.0
        elif margin < 0.020:
            score += 3.0
        elif margin < 0.040:
            score += 1.0

        pred_c = _pred_candidate(r)
        pred_lex = float((pred_c or {}).get("lex") or 0.0)
        has_alt_high_lex = any(float(c.get("lex") or 0.0) >= 0.50 for c in cands if c is not pred_c)
        if pred_lex < 0.20 and has_alt_high_lex:
            score += 2.0

        if base_score < 0.55:
            score += 1.0

        # Prefer to manually check CONC/UNKN predictions and T170 concepts.
        if pred_group in {"CONC", "UNKN"}:
            score += 2.0
        if pred_tui.upper() == "T170":
            score += 2.0

        return score

    def score_bucket(r: Dict[str, Any]) -> str:
        s = float((r.get("extraction") or {}).get("score") or 0.0)
        if s < 0.55:
            return "low"
        if s >= 0.70:
            return "high"
        return "mid"

    # Stratify by note_type first.
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_type.setdefault(str(r.get("note_type") or "Unknown"), []).append(r)

    target_n = max(1, int(args.n))
    per_type = max(1, target_n // max(1, len(by_type)))

    selected: List[Dict[str, Any]] = []
    selected_keys = set()

    def take(cands: List[Dict[str, Any]], n: int) -> None:
        if n <= 0:
            return
        rng.shuffle(cands)
        taken = 0
        for r in cands:
            if len(selected) >= target_n or taken >= n:
                return
            k = _key(r)
            if k in selected_keys:
                continue
            selected_keys.add(k)
            selected.append(r)
            taken += 1

    for note_type, items in sorted(by_type.items(), key=lambda kv: kv[0]):
        # Prefer "tricky" rows (ranking-sensitive, UI/CTCAE-ish terms, low margins), then low-score, then random.
        items_scored = [(float(_tricky_score(r)), r) for r in items]
        # Stable-ish shuffle to avoid always taking the same ties.
        rng.shuffle(items_scored)
        items_scored.sort(key=lambda t: t[0], reverse=True)

        tricky = [r for s, r in items_scored if s >= 5.0]
        low = [r for s, r in items_scored if score_bucket(r) == "low" and r not in tricky]
        rest = [r for _, r in items_scored if r not in tricky and r not in low]

        # Per note type: ~2/3 tricky, ~1/3 fill (low-score then random).
        n_type = int(per_type)
        n_tricky = min(len(tricky), max(0, int(round(n_type * 0.67))))
        n_fill = max(0, n_type - n_tricky)

        take(tricky, n_tricky)
        take(low + rest, n_fill)

    # Fill remaining slots with globally weird + low-score items, then random.
    if len(selected) < target_n:
        remaining = [r for r in rows if _key(r) not in selected_keys]
        remaining_scored = [(float(_tricky_score(r)), r) for r in remaining]
        rng.shuffle(remaining_scored)
        remaining_scored.sort(key=lambda t: t[0], reverse=True)

        fill = [r for s, r in remaining_scored if s >= 5.0]
        fill += [r for s, r in remaining_scored if score_bucket(r) == "low" and r not in fill]
        fill += [r for _, r in remaining_scored if r not in fill]
        take(fill, target_n - len(selected))

    # Add reviewer label scaffold.
    out_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(selected[:target_n]):
        rr = dict(r)
        rr["id"] = i + 1
        rr["label"] = {"gold_cui": "", "notes": ""}
        out_rows.append(rr)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {args.output_jsonl} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
