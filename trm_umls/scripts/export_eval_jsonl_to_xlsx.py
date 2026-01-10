#!/usr/bin/env python3
"""
Export `eval_notes_dir.py` JSONL output to a color-coded XLSX for review.

Input JSONL rows look like:
  {
    "note_file": "...",
    "note_type": "...",
    "threshold": 0.55,
    "context": {"left": "...", "span": "...", "right": "..."},
    "extraction": {... ConceptExtraction as dict ...}
  }

This exporter is intentionally lightweight:
  - Streams JSONL line-by-line (no need to load everything into RAM)
  - Writes an "extractions" sheet + a small "summary" sheet
  - Adds basic color coding:
      * semantic_group cell background
      * assertion cell background
      * score conditional formatting (3-color scale)

Privacy note:
  - The XLSX can contain clinical text in the span/context columns.
    Treat it as sensitive and keep it local.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _group_color(group: str) -> str:
    g = (group or "").strip().upper()
    # Light fills so the sheet stays readable.
    return {
        "DISO": "#FADBD8",  # light red
        "CHEM": "#D6EAF8",  # light blue
        "PROC": "#D5F5E3",  # light green
        "ANAT": "#FCF3CF",  # light yellow
        "PHEN": "#E8DAEF",  # light purple
        "CONC": "#FDEBD0",  # light orange
    }.get(g, "#EAECEE")  # light gray


def _assertion_color(assertion: str) -> str:
    a = (assertion or "").strip().upper()
    return {
        "PRESENT": "#D5F5E3",  # light green
        "ABSENT": "#FADBD8",  # light red
        "POSSIBLE": "#FCF3CF",  # light yellow
    }.get(a, "#FFFFFF")


def main() -> None:
    p = argparse.ArgumentParser(description="Export eval JSONL to XLSX (color-coded)")
    p.add_argument("--eval-jsonl", type=Path, required=True)
    p.add_argument("--output-xlsx", type=Path, required=True)
    p.add_argument("--no-context", action="store_true", help="Omit left/span/right context columns (safer)")
    p.add_argument("--max-rows", type=int, default=0, help="Stop after N rows (0 = all)")
    args = p.parse_args()

    if not args.eval_jsonl.exists():
        raise FileNotFoundError(args.eval_jsonl)
    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    import xlsxwriter

    # Columns are intentionally review-friendly (what people actually scan).
    cols: List[Tuple[str, str]] = [
        ("note_file", "note_file"),
        ("start", "start"),
        ("end", "end"),
        ("span", "span"),
        ("expanded", "expanded_text"),
        ("normalized", "normalized_text"),
        ("cui", "cui"),
        ("preferred_term", "preferred_term"),
        ("tui", "tui"),
        ("semantic_group", "semantic_group"),
        ("semantic_type", "semantic_type"),
        ("assertion", "assertion"),
        ("subject", "subject"),
        ("score", "score"),
        ("severity", "severity"),
        ("laterality", "laterality"),
        ("temporality", "temporality"),
    ]
    if not args.no_context:
        cols.extend(
            [
                ("left_context", "ctx_left"),
                ("right_context", "ctx_right"),
            ]
        )

    wb = xlsxwriter.Workbook(str(args.output_xlsx))
    ws = wb.add_worksheet("extractions")
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, 0, max(0, len(cols) - 1))

    header_fmt = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
    text_fmt = wb.add_format({"text_wrap": True, "valign": "top"})
    score_fmt = wb.add_format({"num_format": "0.000", "valign": "top"})

    group_fmts = {g: wb.add_format({"bg_color": _group_color(g), "valign": "top"}) for g in [
        "ANAT", "CHEM", "CONC", "DISO", "PHEN", "PROC", "UNKN"
    ]}
    default_group_fmt = wb.add_format({"bg_color": _group_color(""), "valign": "top"})

    assertion_fmts = {
        "PRESENT": wb.add_format({"bg_color": _assertion_color("PRESENT"), "valign": "top"}),
        "ABSENT": wb.add_format({"bg_color": _assertion_color("ABSENT"), "valign": "top"}),
        "POSSIBLE": wb.add_format({"bg_color": _assertion_color("POSSIBLE"), "valign": "top"}),
    }
    default_assert_fmt = wb.add_format({"valign": "top"})

    # Write header row.
    for c, (label, _key) in enumerate(cols):
        ws.write(0, c, label, header_fmt)

    # Set column widths (rough defaults; Excel users can adjust).
    widths = {
        "note_file": 24,
        "start": 7,
        "end": 7,
        "span": 36,
        "expanded": 36,
        "normalized": 36,
        "cui": 12,
        "preferred_term": 44,
        "tui": 7,
        "semantic_group": 12,
        "semantic_type": 22,
        "assertion": 10,
        "subject": 10,
        "score": 8,
        "severity": 10,
        "laterality": 10,
        "temporality": 12,
        "left_context": 28,
        "right_context": 28,
    }
    for c, (label, _key) in enumerate(cols):
        ws.set_column(c, c, widths.get(label, 18))

    # Aggregates for the summary sheet.
    total_rows = 0
    by_note = Counter()
    by_group = Counter()
    by_group_assert = Counter()

    # Stream rows.
    r = 1
    max_rows = int(args.max_rows)
    for obj in _iter_jsonl(args.eval_jsonl):
        note_file = _safe_str(obj.get("note_file"))
        ctx = obj.get("context") or {}
        extr = obj.get("extraction") or {}

        start = _safe_int(extr.get("start"))
        end = _safe_int(extr.get("end"))
        span = _safe_str(ctx.get("span") or extr.get("text"))
        expanded = _safe_str(extr.get("expanded_text"))
        normalized = _safe_str(extr.get("normalized_text"))

        cui = _safe_str(extr.get("cui"))
        preferred = _safe_str(extr.get("preferred_term"))
        tui = _safe_str(extr.get("tui"))
        group = _safe_str(extr.get("semantic_group"))
        stype = _safe_str(extr.get("semantic_type"))
        assertion = _safe_str(extr.get("assertion"))
        subject = _safe_str(extr.get("subject"))
        score = _safe_float(extr.get("score"))

        severity = _safe_str(extr.get("severity"))
        laterality = _safe_str(extr.get("laterality"))
        temporality = _safe_str(extr.get("temporality"))

        left = _safe_str(ctx.get("left"))
        right = _safe_str(ctx.get("right"))

        # Update summary counters.
        total_rows += 1
        by_note[note_file] += 1
        by_group[group] += 1
        by_group_assert[(group, assertion)] += 1

        group_fmt = group_fmts.get((group or "").upper(), default_group_fmt)
        assert_fmt = assertion_fmts.get((assertion or "").upper(), default_assert_fmt)

        # Write cells.
        values: Dict[str, Any] = {
            "note_file": note_file,
            "start": start if start is not None else "",
            "end": end if end is not None else "",
            "span": span,
            "expanded_text": expanded,
            "normalized_text": normalized,
            "cui": cui,
            "preferred_term": preferred,
            "tui": tui,
            "semantic_group": group,
            "semantic_type": stype,
            "assertion": assertion,
            "subject": subject,
            "score": score if score is not None else "",
            "severity": severity,
            "laterality": laterality,
            "temporality": temporality,
            "ctx_left": left,
            "ctx_right": right,
        }

        for c, (label, key) in enumerate(cols):
            v = values.get(key, "")
            if label == "score":
                ws.write_number(r, c, float(v) if v != "" else 0.0, score_fmt)
                if v == "":
                    # Keep blanks blank (write_number forces 0). Overwrite with blank.
                    ws.write_blank(r, c, None, score_fmt)
                continue
            if label == "semantic_group":
                ws.write(r, c, v, group_fmt)
                continue
            if label == "assertion":
                ws.write(r, c, v, assert_fmt)
                continue
            ws.write(r, c, v, text_fmt)

        r += 1
        if max_rows > 0 and total_rows >= max_rows:
            break

    # Score column conditional formatting (3-color scale).
    score_col = None
    for idx, (label, _key) in enumerate(cols):
        if label == "score":
            score_col = idx
            break
    if score_col is not None and r > 1:
        ws.conditional_format(1, score_col, r - 1, score_col, {"type": "3_color_scale"})

    # Summary sheet.
    ws2 = wb.add_worksheet("summary")
    ws2.freeze_panes(1, 0)
    ws2.write(0, 0, "metric", header_fmt)
    ws2.write(0, 1, "value", header_fmt)
    ws2.write(1, 0, "eval_jsonl", text_fmt)
    ws2.write(1, 1, str(args.eval_jsonl), text_fmt)
    ws2.write(2, 0, "rows", text_fmt)
    ws2.write_number(2, 1, int(total_rows))
    ws2.write(3, 0, "unique_notes", text_fmt)
    ws2.write_number(3, 1, int(len(by_note)))

    # Group breakdown table.
    row0 = 5
    ws2.write(row0, 0, "semantic_group", header_fmt)
    ws2.write(row0, 1, "rows", header_fmt)
    ws2.write(row0, 2, "present", header_fmt)
    ws2.write(row0, 3, "absent", header_fmt)
    ws2.write(row0, 4, "possible", header_fmt)

    rr = row0 + 1
    for group, cnt in sorted(by_group.items(), key=lambda kv: (-kv[1], kv[0])):
        g = (group or "").upper()
        ws2.write(rr, 0, g, group_fmts.get(g, default_group_fmt))
        ws2.write_number(rr, 1, int(cnt))
        ws2.write_number(rr, 2, int(by_group_assert.get((group, "PRESENT"), 0)))
        ws2.write_number(rr, 3, int(by_group_assert.get((group, "ABSENT"), 0)))
        ws2.write_number(rr, 4, int(by_group_assert.get((group, "POSSIBLE"), 0)))
        rr += 1

    # Per-note counts (top 50 by extractions).
    rr += 2
    ws2.write(rr, 0, "note_file (top 50)", header_fmt)
    ws2.write(rr, 1, "rows", header_fmt)
    rr += 1
    for note_file, cnt in by_note.most_common(50):
        ws2.write(rr, 0, note_file, text_fmt)
        ws2.write_number(rr, 1, int(cnt))
        rr += 1

    wb.close()
    print(f"Wrote {args.output_xlsx}")


if __name__ == "__main__":
    main()

