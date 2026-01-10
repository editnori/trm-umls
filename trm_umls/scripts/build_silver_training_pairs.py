#!/usr/bin/env python3
"""
Build a training-pairs embedding directory by mixing:
  - base UMLS synonym pairs (clean signal)
  - silver span->CUI pairs (weak signal) from teacher labeling + SapBERT linking

This script writes a directory usable by:
  `python trm_umls/train_multitask.py --embeddings-dir <out> --embedding-target concept`

FLOW: build_silver_training_pairs
Entrypoint: `python trm_umls/scripts/build_silver_training_pairs.py ...`
Inputs:
  - `--base-embeddings-dir` containing: `embedding_metadata.json`, `cui_array.npy`, `umls_embeddings.npy`,
    and optionally `synonym_texts.json`, `synonym_cuis.npy`
  - `--silver-jsonl` one or more JSONL files from `link_spans_to_cuis.py`
Happy path:
  1) Load base synonym pairs (optionally sample).
  2) Load silver pairs, filter by quality rules, and de-dupe.
  3) Concatenate base + silver pairs.
  4) Write `synonym_texts.json` + `synonym_cuis.npy` into `--output-dir`.
  5) Symlink (or copy) concept files from `--base-embeddings-dir`.
Outputs:
  - `--output-dir/synonym_texts.json`
  - `--output-dir/synonym_cuis.npy`
  - `--output-dir/embedding_metadata.json` (copied + updated counts)
  - symlinks/copies of `cui_array.npy` and `umls_embeddings.npy`
Side effects:
  - File writes in `--output-dir`.
Failure modes:
  - Missing required base files -> exit non-zero.
  - Invalid JSON/NPY -> exception.
Observability:
  - Prints counts and sampling decisions.

Invariants:
  - `len(synonym_texts) == len(synonym_cuis)` in the output.
  - Silver pairs always have a valid `Cxxxxxxx` CUI.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_paths(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".jsonl"]))
        else:
            out.append(p)
    return out


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        # Fallback: copy if symlink is not allowed.
        shutil.copyfile(src, dst)


def _cui_to_int(cui: str) -> Optional[int]:
    c = (cui or "").strip()
    if not c:
        return None
    if c.startswith("C"):
        c = c[1:]
    try:
        return int(c)
    except ValueError:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Build mixed (base + silver) training pairs for TRM concept-mode training")
    p.add_argument("--base-embeddings-dir", type=Path, default=Path("trm_umls") / "data" / "embeddings")
    p.add_argument("--silver-jsonl", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--max-base", type=int, default=200_000, help="Sample at most N base synonym pairs (0 = use all)")
    p.add_argument("--max-silver", type=int, default=200_000, help="Keep at most N silver pairs (0 = use all)")
    p.add_argument("--seed", type=int, default=13)

    p.add_argument("--min-sim", type=float, default=0.65)
    p.add_argument("--min-lex", type=float, default=0.15)
    p.add_argument("--min-teacher-count", type=int, default=2)
    p.add_argument("--min-chars", type=int, default=2)
    p.add_argument("--max-chars", type=int, default=120)
    p.add_argument("--dedupe", action="store_true", help="De-dupe silver pairs by (text_lower, cui)")
    args = p.parse_args()

    rng = random.Random(int(args.seed))

    base_dir = Path(args.base_embeddings_dir)
    meta_path = base_dir / "embedding_metadata.json"
    cui_array_path = base_dir / "cui_array.npy"
    concept_emb_path = base_dir / "umls_embeddings.npy"

    if not meta_path.exists():
        raise SystemExit(f"Missing {meta_path}")
    if not cui_array_path.exists():
        raise SystemExit(f"Missing {cui_array_path}")
    if not concept_emb_path.exists():
        raise SystemExit(f"Missing {concept_emb_path}")

    base_syn_texts_path = base_dir / "synonym_texts.json"
    base_syn_cuis_path = base_dir / "synonym_cuis.npy"
    if not base_syn_texts_path.exists() or not base_syn_cuis_path.exists():
        raise SystemExit(f"Missing base synonym files: {base_syn_texts_path} and/or {base_syn_cuis_path}")

    base_texts: List[str] = json.loads(base_syn_texts_path.read_text(encoding="utf-8"))
    base_cuis = np.load(base_syn_cuis_path, mmap_mode="r")
    if len(base_texts) != int(base_cuis.shape[0]):
        raise SystemExit("Base synonym_texts.json and synonym_cuis.npy length mismatch")

    base_n = len(base_texts)
    max_base = int(args.max_base)
    if max_base > 0 and base_n > max_base:
        idxs = rng.sample(range(base_n), max_base)
        idxs.sort()
        base_texts = [base_texts[i] for i in idxs]
        base_cuis = np.asarray(base_cuis[idxs], dtype=np.int64)
        print(f"[base] sampled {len(base_texts):,}/{base_n:,}")
    else:
        base_cuis = np.asarray(base_cuis, dtype=np.int64)
        print(f"[base] using all {base_n:,}")

    silver_paths = _iter_paths(args.silver_jsonl)
    if not silver_paths:
        raise SystemExit("No silver JSONL files found via --silver-jsonl")

    min_chars = int(args.min_chars)
    max_chars = int(args.max_chars)
    min_sim = float(args.min_sim)
    min_lex = float(args.min_lex)
    min_teacher = int(args.min_teacher_count)

    silver_texts: List[str] = []
    silver_cuis: List[int] = []
    seen: set[Tuple[str, int]] = set()

    total_rows = 0
    kept_rows = 0
    for sp in silver_paths:
        for row in _load_jsonl(sp):
            total_rows += 1
            cui = str(row.get("cui") or "").strip()
            cui_int = _cui_to_int(cui)
            if cui_int is None:
                continue
            try:
                sim = float(row.get("sim") or 0.0)
                lex = float(row.get("lex") or 0.0)
            except Exception:
                sim = 0.0
                lex = 0.0
            if sim < min_sim or lex < min_lex:
                continue
            try:
                teacher_count = int(row.get("teacher_count") or 0)
            except Exception:
                teacher_count = 0
            if teacher_count and teacher_count < min_teacher:
                continue

            text = str(row.get("normalized") or row.get("text") or "").strip()
            if not text:
                continue
            if len(text) < min_chars or len(text) > max_chars:
                continue

            if args.dedupe:
                k = (text.lower(), int(cui_int))
                if k in seen:
                    continue
                seen.add(k)

            silver_texts.append(text)
            silver_cuis.append(int(cui_int))
            kept_rows += 1

            if int(args.max_silver) > 0 and len(silver_texts) >= int(args.max_silver):
                break
        if int(args.max_silver) > 0 and len(silver_texts) >= int(args.max_silver):
            break

    print(f"[silver] kept {len(silver_texts):,}/{total_rows:,} (min_sim={min_sim} min_lex={min_lex} min_teacher={min_teacher})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Link or copy concept universe files.
    shutil.copyfile(meta_path, out_dir / "embedding_metadata.json")
    _safe_symlink_or_copy(cui_array_path, out_dir / "cui_array.npy")
    _safe_symlink_or_copy(concept_emb_path, out_dir / "umls_embeddings.npy")

    # Write combined training pairs.
    texts = list(base_texts) + list(silver_texts)
    cuis = np.concatenate([np.asarray(base_cuis, dtype=np.int64), np.asarray(silver_cuis, dtype=np.int64)], axis=0)
    if len(texts) != int(cuis.shape[0]):
        raise SystemExit("Output texts/cuis length mismatch (bug)")

    (out_dir / "synonym_texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    np.save(out_dir / "synonym_cuis.npy", cuis)

    # Update metadata counts (best-effort; keep original prefterms/cuis).
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["num_synonyms"] = int(len(texts))
        meta["silver_synonyms"] = int(len(silver_texts))
        (out_dir / "embedding_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    print(f"Wrote {out_dir}")
    print(f"  pairs_total: {len(texts):,}")
    print(f"    base: {len(base_texts):,}")
    print(f"    silver: {len(silver_texts):,}")


if __name__ == "__main__":
    main()
