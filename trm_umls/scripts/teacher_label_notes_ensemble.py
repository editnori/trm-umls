#!/usr/bin/env python3
"""
Multi-teacher span labeling for clinical notes.

This produces "teacher spans" (start/end/text) that can later be linked to UMLS
and distilled into the TRM model.

FLOW: teacher_label_notes_ensemble
Entrypoint: `python trm_umls/scripts/teacher_label_notes_ensemble.py ...`
Inputs:
  - `--notes-dir`: directory of `.txt` notes
  - `--ner-model`: one or more HF token-classification models
  - `--use-abbrev-teacher`: optional abbreviation span teacher
Happy path:
  1) Read notes (bounded by `--max-bytes`) in batches.
  2) Run each NER teacher over the batch.
  3) Optionally add abbreviation spans (HTN, COPD, ...).
  4) Merge overlapping spans into clusters and record teacher agreement.
  5) Keep only clusters that meet `min_teachers` OR a single-teacher high-conf rule.
  6) Write one JSONL row per note with its merged mentions.
Outputs:
  - JSONL at `--output-jsonl` with per-note `mentions[]`.
  - Log file + resumable progress JSON.
Side effects:
  - Writes output and progress to disk.
Failure modes:
  - No notes found -> exit non-zero.
  - Model download/load failure -> exception.
  - CUDA OOM -> exception.
Observability:
  - Periodic `processed=...` logs and `*.progress.json`.

Invariants:
  - Emitted mentions have `0 <= start < end <= len(note_text)` and non-empty `text`.
  - `teacher_count == len(teachers)` and `teacher_count >= 1` for every mention.
  - Mentions shorter than `--short-span-min-chars` are emitted only if allowlisted.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _setup_logger(*, log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("teacher_label_ensemble")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _iter_note_files(notes_dir: Path, limit: Optional[int]) -> List[Path]:
    files = sorted([p for p in notes_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    if limit is not None:
        files = files[: max(0, int(limit))]
    return files


def _device_arg(device: str) -> int:
    d = str(device or "").strip().lower()
    if d in {"cpu", "-1"}:
        return -1
    return 0


def _overlap_ratio(a0: int, a1: int, b0: int, b1: int) -> float:
    a0, a1 = int(a0), int(a1)
    b0, b1 = int(b0), int(b1)
    if a1 <= a0 or b1 <= b0:
        return 0.0
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    denom = max(1, min(a1 - a0, b1 - b0))
    return float(inter) / float(denom)


def _normalize_header(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"\s*/\s*", "/", t)
    t = t.rstrip(":")
    t = re.sub(r"\s+", " ", t)
    return t


def _clean_alnum(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", (text or "")).strip()


@dataclass
class Mention:
    start: int
    end: int
    text: str
    normalized: str
    label: str
    score: float
    teacher: str


@dataclass
class MergedMention:
    start: int
    end: int
    text: str
    normalized: str
    label: str
    score: float
    teachers: List[str]
    teacher_count: int
    max_score: float
    mean_score: float


def _postprocess_mentions(
    mentions: List[Mention],
    *,
    min_score: float,
    text_len: int,
) -> List[Mention]:
    out: List[Mention] = []
    seen: set[Tuple[int, int, str, str]] = set()
    for m in mentions:
        if float(m.score) < float(min_score):
            continue
        s = int(m.start)
        e = int(m.end)
        if s < 0 or e <= s or e > int(text_len):
            continue
        text = str(m.text or "").strip()
        if not text:
            continue
        key = (s, e, text.lower(), str(m.teacher))
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _merge_mentions(
    mentions: List[Mention],
    *,
    note_text: str,
    overlap_threshold: float,
    agreement_bonus: float,
) -> List[MergedMention]:
    if not mentions:
        return []

    mentions = sorted(mentions, key=lambda m: (int(m.start), int(m.end), -float(m.score), m.teacher))
    clusters: List[List[Mention]] = []

    for m in mentions:
        placed = False
        for c in clusters:
            # Compare to the current representative (highest score, shortest span).
            rep = c[0]
            if _overlap_ratio(m.start, m.end, rep.start, rep.end) >= float(overlap_threshold):
                c.append(m)
                c.sort(key=lambda x: (-float(x.score), int(x.end) - int(x.start), int(x.start)))
                placed = True
                break
        if not placed:
            clusters.append([m])

    merged: List[MergedMention] = []
    for c in clusters:
        # Representative mention: highest score, then shortest span.
        c_sorted = sorted(c, key=lambda m: (-float(m.score), int(m.end) - int(m.start), int(m.start)))
        rep = c_sorted[0]
        s = int(rep.start)
        e = int(rep.end)
        raw_span = note_text[s:e] if 0 <= s < e <= len(note_text) else rep.text
        teachers = sorted({m.teacher for m in c})
        teacher_count = len(teachers)
        scores = [float(m.score) for m in c]
        max_score = max(scores) if scores else 0.0
        mean_score = float(sum(scores) / max(1, len(scores)))

        agg_score = float(max_score + float(agreement_bonus) * max(0, teacher_count - 1))
        agg_score = float(min(1.0, max(0.0, agg_score)))

        # Best-effort label: choose the label from the representative.
        label = str(rep.label or "").strip()

        merged.append(
            MergedMention(
                start=s,
                end=e,
                text=str(raw_span),
                normalized=str(rep.normalized or raw_span).strip(),
                label=label,
                score=agg_score,
                teachers=teachers,
                teacher_count=int(teacher_count),
                max_score=float(max_score),
                mean_score=float(mean_score),
            )
        )

    # Stable output ordering.
    merged.sort(key=lambda m: (int(m.start), int(m.end), -float(m.score)))
    return merged


def _abbrev_mentions(
    text: str,
    *,
    teacher_name: str,
    allowlist: set[str],
    abbrev_map: Optional[Dict[str, str]] = None,
    short_span_min_chars: int,
) -> List[Mention]:
    # Keep this small and predictable: only match allowlisted abbreviations.
    if not text or not allowlist:
        return []
    # Word-ish boundaries for alnum abbreviations.
    pat = re.compile(r"\b([A-Za-z0-9]{2,12})\b")
    out: List[Mention] = []
    for m in pat.finditer(text):
        tok = m.group(1)
        if not tok:
            continue
        # Reduce false positives: require caps or digits (notes often have lowercase noise).
        if not any(ch.isupper() for ch in tok) and not any(ch.isdigit() for ch in tok):
            continue
        up = tok.upper()
        if up not in allowlist:
            continue
        if len(_clean_alnum(tok)) < int(short_span_min_chars):
            # Allowlist controls short spans; still enforce exact membership.
            pass
        norm = tok
        if abbrev_map is not None:
            norm = str(abbrev_map.get(up) or tok)
        out.append(
            Mention(
                start=int(m.start(1)),
                end=int(m.end(1)),
                text=tok,
                normalized=norm,
                label="ABBR",
                score=1.0,
                teacher=str(teacher_name),
            )
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-teacher span labeling (NER ensemble + abbrev)")
    p.add_argument("--notes-dir", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)

    p.add_argument("--ner-model", type=str, action="append", default=[])
    p.add_argument("--ner-teacher-name", type=str, action="append", default=[])
    p.add_argument("--ner-min-score", type=float, action="append", default=[])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--max-bytes", type=int, default=250_000)
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--aggregation", type=str, default="simple")

    p.add_argument("--use-abbrev-teacher", action="store_true")
    p.add_argument(
        "--abbrev-allowlist-json",
        type=Path,
        default=Path("trm_umls") / "data" / "abbreviations.json",
        help="JSON allowlist of abbreviations to emit as spans (keys if object).",
    )
    p.add_argument("--abbrev-teacher-name", type=str, default="abbrev")

    p.add_argument(
        "--section-stoplist",
        type=str,
        default=(
            "assessment,assessment/plan,assessment and plan,a/p,plan,diagnosis,diagnoses,"
            "history,past medical history,pmh,hpi,ros,review of systems,medications,meds,"
            "allergies,family history,social history,physical exam,physical examination,exam,"
            "labs,imaging,problem list,chief complaint,cc,disposition,consults,results,skin,impression"
        ),
    )
    p.add_argument("--short-span-min-chars", type=int, default=4)
    p.add_argument("--short-span-allowlist", type=str, default="EGD,CABG,HGB,LHC,HTN,DM,CHF,COPD")

    p.add_argument("--overlap-threshold", type=float, default=0.80)
    p.add_argument("--min-teachers", type=int, default=2)
    p.add_argument("--high-conf-single", type=float, default=0.90)
    p.add_argument("--agreement-bonus", type=float, default=0.08)

    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ner_models = [str(m).strip() for m in (args.ner_model or []) if str(m).strip()]
    if not ner_models:
        ner_models = ["d4data/biomedical-ner-all", "nlpie/clinical-distilbert-i2b2-2010"]

    ner_teacher_names = [str(x).strip() for x in (args.ner_teacher_name or []) if str(x).strip()]
    if ner_teacher_names and len(ner_teacher_names) != len(ner_models):
        raise SystemExit("--ner-teacher-name must match --ner-model count (or omit to auto-name)")
    if not ner_teacher_names:
        ner_teacher_names = [f"ner{i+1}" for i in range(len(ner_models))]

    ner_min_scores = [float(x) for x in (args.ner_min_score or [])]
    if ner_min_scores and len(ner_min_scores) != len(ner_models):
        raise SystemExit("--ner-min-score must match --ner-model count (or omit to use default)")
    if not ner_min_scores:
        ner_min_scores = [0.20 for _ in range(len(ner_models))]

    log_file = args.log_file or (Path("trm_umls") / "runs" / f"teacher_label_ensemble_{_now_tag()}.log")
    progress_file = log_file.with_suffix(log_file.suffix + ".progress.json")
    logger = _setup_logger(log_file=log_file)

    logger.info("Starting ensemble teacher labeling")
    logger.info("  notes_dir=%s", args.notes_dir)
    logger.info("  output_jsonl=%s", args.output_jsonl)
    logger.info("  models=%s", ", ".join(ner_models))
    logger.info("  batch_size=%s stride=%s max_length=%s", args.batch_size, args.stride, args.max_length)
    logger.info("  min_teachers=%s high_conf_single=%s overlap=%.2f", args.min_teachers, args.high_conf_single, args.overlap_threshold)
    logger.info("  log_file=%s", log_file)

    section_headers = {
        _normalize_header(x)
        for x in (args.section_stoplist or "").split(",")
        if _normalize_header(x)
    }
    short_allow = {x.strip().upper() for x in (args.short_span_allowlist or "").split(",") if x.strip()}

    abbrev_allow: set[str] = set()
    abbrev_map: Optional[Dict[str, str]] = None
    if bool(args.use_abbrev_teacher):
        try:
            raw = json.loads(Path(args.abbrev_allowlist_json).read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                abbrev_map = {str(k).strip().upper(): str(v).strip() for k, v in raw.items() if str(k).strip()}
                abbrev_allow = set(abbrev_map.keys())
            elif isinstance(raw, list):
                abbrev_allow = {str(k).strip().upper() for k in raw if str(k).strip()}
                abbrev_map = None
            else:
                raise ValueError("expected JSON object or list")
        except Exception as e:
            raise SystemExit(f"Failed to load abbrev allowlist: {e}") from e

    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    teachers: List[Tuple[str, Any, float]] = []
    for model_name, teacher_name, min_score in zip(ner_models, ner_teacher_names, ner_min_scores):
        logger.info("Loading NER teacher: %s (%s)", teacher_name, model_name)
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if int(args.max_length) > 0:
            tok.model_max_length = int(args.max_length)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        ner_pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tok,
            aggregation_strategy=str(args.aggregation),
            device=_device_arg(args.device),
        )
        teachers.append((teacher_name, ner_pipe, float(min_score)))

    note_paths = _iter_note_files(args.notes_dir, args.limit_files)
    if not note_paths:
        raise SystemExit(f"No .txt files in {args.notes_dir}")

    done: set[str] = set()
    if args.output_jsonl.exists() and not args.overwrite:
        try:
            with args.output_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        nf = str(row.get("note_file") or "").strip()
                        if nf:
                            done.add(nf)
                    except Exception:
                        continue
        except Exception:
            done = set()

    to_process = [p for p in note_paths if (p.name not in done) or args.overwrite]
    logger.info("notes_total=%s already_done=%s to_process=%s", len(note_paths), len(done), len(to_process))
    if not to_process:
        logger.info("Nothing to process, exiting")
        return

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else ("a" if args.output_jsonl.exists() else "w")
    out_f = args.output_jsonl.open(mode, encoding="utf-8")

    processed = 0
    t0 = time.time()
    next_log = max(1, int(args.log_every))

    def _write_progress() -> None:
        elapsed_s = time.time() - t0
        payload = {
            "notes_dir": str(args.notes_dir),
            "output_jsonl": str(args.output_jsonl),
            "notes_total": len(note_paths),
            "notes_done": processed,
            "elapsed_s": float(elapsed_s),
            "models": ner_models,
            "teachers": ner_teacher_names + ([str(args.abbrev_teacher_name)] if args.use_abbrev_teacher else []),
            "min_teachers": int(args.min_teachers),
            "high_conf_single": float(args.high_conf_single),
            "overlap_threshold": float(args.overlap_threshold),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _atomic_write_json(progress_file, payload)

    for batch_start in range(0, len(to_process), int(args.batch_size)):
        batch_paths = to_process[batch_start : batch_start + int(args.batch_size)]
        batch_texts: List[str] = []
        batch_meta: List[Dict[str, Any]] = []
        for note_path in batch_paths:
            raw_bytes = note_path.read_bytes()[: max(0, int(args.max_bytes))]
            text = raw_bytes.decode("utf-8", errors="ignore")
            batch_texts.append(text)
            batch_meta.append({"path": note_path, "bytes_read": int(len(raw_bytes))})

        # Run all NER teachers on this batch.
        per_teacher: Dict[str, List[List[Dict[str, Any]]]] = {}
        for teacher_name, ner_pipe, _ in teachers:
            ents = ner_pipe(
                batch_texts,
                batch_size=int(args.batch_size),
                stride=int(args.stride),
            )
            per_teacher[str(teacher_name)] = ents

        for i, meta in enumerate(batch_meta):
            note_path = meta["path"]
            note_text = batch_texts[i]

            raw_mentions: List[Mention] = []
            for teacher_name, ents in per_teacher.items():
                for ent in (ents[i] or []):
                    try:
                        label = str(ent.get("entity_group") or ent.get("entity") or "").strip()
                        word = str(ent.get("word") or "").strip()
                        s = int(ent.get("start", -1))
                        e = int(ent.get("end", -1))
                        raw_mentions.append(
                            Mention(
                                start=s,
                                end=e,
                                text=word,
                                normalized=word,
                                label=label,
                                score=float(ent.get("score") or 0.0),
                                teacher=str(teacher_name),
                            )
                        )
                    except Exception:
                        continue

            if args.use_abbrev_teacher:
                for m in _abbrev_mentions(
                    note_text,
                    teacher_name=str(args.abbrev_teacher_name),
                    allowlist=abbrev_allow,
                    abbrev_map=abbrev_map,
                    short_span_min_chars=int(args.short_span_min_chars),
                ):
                    raw_mentions.append(m)

            # Per-teacher score thresholds + bounds.
            mentions_pp: List[Mention] = []
            for teacher_name, _, teacher_min in teachers:
                ms = [m for m in raw_mentions if m.teacher == str(teacher_name)]
                mentions_pp.extend(_postprocess_mentions(ms, min_score=float(teacher_min), text_len=len(note_text)))
            if args.use_abbrev_teacher:
                ms = [m for m in raw_mentions if m.teacher == str(args.abbrev_teacher_name)]
                mentions_pp.extend(_postprocess_mentions(ms, min_score=0.0, text_len=len(note_text)))

            merged = _merge_mentions(
                mentions_pp,
                note_text=note_text,
                overlap_threshold=float(args.overlap_threshold),
                agreement_bonus=float(args.agreement_bonus),
            )

            kept: List[Dict[str, Any]] = []
            for mm in merged:
                raw = str(mm.text or "").strip()
                norm_header = _normalize_header(raw)
                if norm_header and norm_header in section_headers:
                    continue

                cleaned = _clean_alnum(raw)
                if not cleaned:
                    continue
                if len(cleaned) < int(args.short_span_min_chars) and cleaned.upper() not in short_allow:
                    continue

                if int(mm.teacher_count) >= int(args.min_teachers) or float(mm.max_score) >= float(args.high_conf_single):
                    kept.append(
                        {
                            "text": str(mm.text),
                            "normalized": str(mm.normalized),
                            "start": int(mm.start),
                            "end": int(mm.end),
                            "label": str(mm.label),
                            "score": float(mm.score),
                            "teacher_count": int(mm.teacher_count),
                            "teachers": list(mm.teachers),
                            "max_score": float(mm.max_score),
                            "mean_score": float(mm.mean_score),
                        }
                    )

            row = {
                "note_file": note_path.name,
                "note_type": note_path.stem,
                "bytes_read": int(meta.get("bytes_read", 0)),
                "mentions": kept,
                "stats": {
                    "mentions_raw": int(len(raw_mentions)),
                    "mentions_post": int(len(mentions_pp)),
                    "mentions_merged": int(len(merged)),
                    "mentions_kept": int(len(kept)),
                },
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            processed += 1

        if processed >= next_log:
            elapsed = time.time() - t0
            notes_per_s = processed / elapsed if elapsed > 0 else 0.0
            logger.info("processed=%s/%s notes_per_s=%.2f", processed, len(note_paths), notes_per_s)
            _write_progress()
            next_log += max(1, int(args.log_every))

    _write_progress()
    out_f.close()
    logger.info("DONE processed=%s", processed)


if __name__ == "__main__":
    main()
