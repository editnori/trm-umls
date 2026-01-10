#!/usr/bin/env python3
"""
Evaluate concept extraction on a directory of note files.

Writes JSONL with mention-level predictions (spans + before/after).

This is intended for manual review / benchmarking. Keep outputs local.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Ensure repo root is on sys.path so `import trm_umls` works even when executed by file path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trm_umls.pipeline import TRMUMLSPipeline


def _cut_after_n_sentences(text: str, max_sentences: Optional[int]) -> str:
    if max_sentences is None or max_sentences <= 0:
        return text
    n = 0
    cut = len(text)
    for m in re.finditer(r"[.;!\n]+", text):
        n += 1
        if n >= max_sentences:
            cut = m.end()
            break
    return text[:cut]


def _iter_note_files(notes_dir: Path, limit: Optional[int]) -> Iterable[Path]:
    files = sorted([p for p in notes_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    if limit is not None:
        files = files[: max(0, int(limit))]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TRM-UMLS extraction on a notes directory (JSONL)")
    parser.add_argument("--notes-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--tui-mappings-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=10, help="FAISS top-k candidates per mention")
    parser.add_argument("--no-rerank", action="store_true", help="Disable lexical reranking when top-k > 1")
    parser.add_argument("--lexical-weight", type=float, default=0.30, help="Rerank weight for token overlap")
    parser.add_argument(
        "--rerank-margin",
        type=float,
        default=0.04,
        help="Only allow rerank to switch within this score delta of the top FAISS hit (set <0 to disable).",
    )
    parser.add_argument("--relation-rerank", action="store_true", help="Enable MRREL relation-based reranking (optional)")
    parser.add_argument("--relation-weight", type=float, default=0.05, help="Rerank weight per MRREL neighbor hit")
    parser.add_argument("--relation-max-degree", type=int, default=2000, help="Skip relation scoring for hub nodes above this degree")
    parser.add_argument("--clinical-rerank", action="store_true", help="Enable clinical rerank biases (semantic group/TUI + prefterm penalties)")
    parser.add_argument("--group-bias", type=str, default="", help="Comma-separated group=weight (e.g., DISO=0.01,PROC=0.005)")
    parser.add_argument("--tui-bias", type=str, default="", help="Comma-separated TUI=weight (e.g., T184=0.01,T033=-0.005)")
    parser.add_argument("--include-candidates", action="store_true", help="Include top-k candidate list per mention (larger JSONL)")
    parser.add_argument("--context-chars", type=int, default=80, help="Chars of left/right context to include per mention")
    parser.add_argument("--max-sentences", type=int, default=120)
    parser.add_argument("--max-bytes", type=int, default=300_000)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--use-model-labels", action="store_true")
    parser.add_argument("--dedupe", action="store_true", help="De-dupe by (CUI, assertion, subject)")
    parser.add_argument(
        "--filter-groups",
        type=str,
        default="ANAT,CHEM,DISO,PHEN,PROC",
        help="Comma-separated semantic groups to keep (empty = keep all)",
    )
    args = parser.parse_args()

    allowed_groups: Optional[Set[str]] = None
    raw_groups = [g.strip().upper() for g in args.filter_groups.split(",") if g.strip()]
    if raw_groups:
        allowed_groups = set(raw_groups)

    pipe = TRMUMLSPipeline.load(
        str(args.checkpoint),
        index_path=str(args.index_path) if args.index_path is not None else None,
        metadata_path=str(args.metadata_path) if args.metadata_path is not None else None,
        tui_mappings_path=str(args.tui_mappings_path) if args.tui_mappings_path is not None else None,
        device=args.device,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}
    total_rows = 0

    with open(args.output_jsonl, "w") as out_f:
        for path in _iter_note_files(args.notes_dir, args.limit_files):
            raw_bytes = path.read_bytes()[: max(0, int(args.max_bytes))]
            text_full = raw_bytes.decode("utf-8", errors="ignore")
            text_used = _cut_after_n_sentences(text_full, args.max_sentences)

            results = pipe.extract(
                text_used,
                threshold=float(args.threshold),
                extract_all=True,
                use_model_labels=bool(args.use_model_labels),
                dedupe=bool(args.dedupe),
                top_k=int(args.top_k),
                rerank=(not bool(args.no_rerank)),
                lexical_weight=float(args.lexical_weight),
                rerank_margin=float(args.rerank_margin),
                relation_rerank=bool(args.relation_rerank),
                relation_weight=float(args.relation_weight),
                relation_max_degree=int(args.relation_max_degree),
                include_candidates=bool(args.include_candidates),
                clinical_rerank=bool(args.clinical_rerank),
                group_bias=TRMUMLSPipeline._parse_bias_map(str(args.group_bias), kind="group"),
                tui_bias=TRMUMLSPipeline._parse_bias_map(str(args.tui_bias), kind="tui"),
            )
            if allowed_groups is not None:
                results = [r for r in results if r.semantic_group in allowed_groups]

            summary[path.name] = {
                "bytes_read": int(len(raw_bytes)),
                "chars_used": int(len(text_used)),
                "extractions": int(len(results)),
            }

            for r in results:
                start = int(getattr(r, "start", -1))
                end = int(getattr(r, "end", -1))
                ctx_n = max(0, int(args.context_chars))
                left = text_used[max(0, start - ctx_n) : start] if start >= 0 else ""
                span = text_used[start:end] if start >= 0 and end >= 0 else ""
                right = text_used[end : end + ctx_n] if start >= 0 and end >= 0 else ""
                row = {
                    "note_file": path.name,
                    "note_type": path.stem,
                    "threshold": float(args.threshold),
                    "context": {"left": left, "span": span, "right": right},
                    "extraction": asdict(r),
                }
                out_f.write(json.dumps(row) + "\n")
                total_rows += 1

    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"files": summary, "total_rows": total_rows}, indent=2))
    print(f"Wrote {args.output_jsonl} ({total_rows} rows)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
