#!/usr/bin/env python3
"""
Extract a *subset* of UMLS from an RRF release (MRCONSO/MRSTY).

Why
  - Our original pipeline used a smaller HSQL dump (`KidneyStone_SDOH.script`).
  - The full UMLS 2025AA RRF (MRCONSO/MRSTY) contains richer synonyms + semantic types.
  - We keep the *concept universe stable* by restricting to the CUIs already present in
    `trm_umls/data/embeddings/cui_array.npy`.

Outputs
  - Updates `trm_umls/data/umls/tui_mappings.json` using MRSTY.RRF (for our CUIs).
  - Updates `trm_umls/data/embeddings/synonym_texts.json` and
    `trm_umls/data/embeddings/synonym_cuis.npy` using MRCONSO.RRF (for our CUIs).

This keeps downstream code simple:
  - Training consumes `synonym_texts.json` + `synonym_cuis.npy`.
  - Inference consumes the concept index + `tui_mappings.json`.

Notes
  - This script is intended to run locally (RRFs are large).
  - It streams writes (no huge in-memory JSON arrays).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


def _cui_to_int(cui: str) -> Optional[int]:
    cui = (cui or "").strip()
    if not cui:
        return None
    if cui.startswith("C") and len(cui) > 1:
        cui = cui[1:]
    try:
        return int(cui)
    except ValueError:
        return None


def _tui_to_int(tui: str) -> Optional[int]:
    tui = (tui or "").strip()
    if not tui:
        return None
    if tui.startswith("T") and len(tui) > 1:
        tui = tui[1:]
    try:
        return int(tui)
    except ValueError:
        return None


def _atomic_replace(tmp_path: Path, final_path: Path) -> None:
    tmp_path.replace(final_path)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _iter_lines(path: Path) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line


def _write_json_array_stream(path_tmp: Path, items: Iterable[str]) -> int:
    """
    Stream-write a JSON array of strings.
    Returns count of written items.
    """
    n = 0
    first = True
    with open(path_tmp, "w", encoding="utf-8") as f:
        f.write("[")
        for item in items:
            if first:
                first = False
            else:
                f.write(",")
            f.write(json.dumps(item, ensure_ascii=False))
            n += 1
        f.write("]\n")
    return n


def main() -> None:
    trm_umls_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Extract UMLS subset from RRF release (MRCONSO/MRSTY)")
    parser.add_argument("--rrf-dir", type=Path, required=True, help="Directory containing MRCONSO.RRF and MRSTY.RRF")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=trm_umls_dir / "data" / "embeddings",
        help="Embeddings dir that contains cui_array.npy and will receive synonym_texts.json/synonym_cuis.npy",
    )
    parser.add_argument(
        "--umls-dir",
        type=Path,
        default=trm_umls_dir / "data" / "umls",
        help="UMLS dir that will receive tui_mappings.json",
    )
    parser.add_argument(
        "--max-synonyms-per-cui",
        type=int,
        default=5,
        help="Max MRCONSO strings to emit per CUI (keeps training set size reasonable)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=3,
        help="Minimum synonym string length to include",
    )
    parser.add_argument(
        "--allowed-sabs",
        type=str,
        default="SNOMEDCT_US,RXNORM,LOINC,ICD10CM,ICD9CM,MSH",
        help="Comma-separated SABs to include (empty = include all)",
    )
    parser.add_argument(
        "--allowed-ttys",
        type=str,
        default="PT,SY,PN,ET,MH",
        help="Comma-separated TTYs to include (empty = include all)",
    )
    parser.add_argument(
        "--suppress",
        type=str,
        default="N",
        help="SUPPRESS value to include from MRCONSO (typically 'N')",
    )
    args = parser.parse_args()

    mrconso = args.rrf_dir / "MRCONSO.RRF"
    mrsty = args.rrf_dir / "MRSTY.RRF"
    if not mrconso.exists():
        raise FileNotFoundError(mrconso)
    if not mrsty.exists():
        raise FileNotFoundError(mrsty)

    cui_array_path = args.embeddings_dir / "cui_array.npy"
    if not cui_array_path.exists():
        raise FileNotFoundError(f"Missing concept list: {cui_array_path}")

    print(f"[{_now()}] Loading concept CUI universe: {cui_array_path}", flush=True)
    concept_cuis = np.load(cui_array_path, mmap_mode="r")
    max_cui = int(concept_cuis.max())
    present = np.zeros(max_cui + 1, dtype=np.uint8)
    present[concept_cuis.astype(np.int64, copy=False)] = 1
    print(f"[{_now()}] Concepts: {int(concept_cuis.shape[0]):,} | max_cui={max_cui:,}", flush=True)

    # 1) MRSTY -> tui_mappings.json
    print(f"[{_now()}] Parsing {mrsty.name} for TUIs...", flush=True)
    tui_map: Dict[int, List[int]] = defaultdict(list)
    line_n = 0
    kept = 0
    for line in _iter_lines(mrsty):
        line_n += 1
        parts = line.rstrip("\n").split("|")
        if len(parts) < 2:
            continue
        cui_int = _cui_to_int(parts[0])
        if cui_int is None or cui_int > max_cui or present[cui_int] == 0:
            continue
        tui_int = _tui_to_int(parts[1])
        if tui_int is None:
            continue
        lst = tui_map[cui_int]
        if tui_int not in lst:
            lst.append(tui_int)
        kept += 1
        if line_n % 2_000_000 == 0:
            print(f"  MRSTY lines={line_n:,} kept_rows={kept:,} mapped_cuis={len(tui_map):,}", flush=True)

    args.umls_dir.mkdir(parents=True, exist_ok=True)
    tui_path = args.umls_dir / "tui_mappings.json"
    tui_tmp = tui_path.with_suffix(".json.tmp")
    with open(tui_tmp, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in tui_map.items()}, f)
    _atomic_replace(tui_tmp, tui_path)
    print(f"[{_now()}] Wrote {tui_path} (cuis={len(tui_map):,})", flush=True)

    # 2) MRCONSO -> synonym_texts.json + synonym_cuis.npy (training pairs)
    allowed_sabs: Optional[Set[str]] = None
    sabs = [s.strip() for s in str(args.allowed_sabs).split(",") if s.strip()]
    if sabs:
        allowed_sabs = set(sabs)

    allowed_ttys: Optional[Set[str]] = None
    ttys = [t.strip() for t in str(args.allowed_ttys).split(",") if t.strip()]
    if ttys:
        allowed_ttys = set(ttys)

    max_per = int(args.max_synonyms_per_cui)
    if max_per <= 0 or max_per > 255:
        raise ValueError("--max-synonyms-per-cui must be in [1, 255]")

    counts = np.zeros(max_cui + 1, dtype=np.uint8)
    train_cuis: List[int] = []

    syn_texts_path = args.embeddings_dir / "synonym_texts.json"
    syn_cuis_path = args.embeddings_dir / "synonym_cuis.npy"
    syn_texts_tmp = syn_texts_path.with_suffix(".json.tmp")
    # Use a tmp filename that still ends with `.npy` so `np.save()` doesn't auto-append.
    syn_cuis_tmp = syn_cuis_path.with_name(syn_cuis_path.name + ".tmp.npy")

    print(f"[{_now()}] Parsing {mrconso.name} for synonym training pairs...", flush=True)
    print(
        f"  filters: LAT=ENG SUPPRESS={args.suppress!r} max_per_cui={max_per} "
        f"SABs={'ALL' if allowed_sabs is None else len(allowed_sabs)} "
        f"TTYs={'ALL' if allowed_ttys is None else len(allowed_ttys)}",
        flush=True,
    )

    line_n = 0
    kept = 0
    first = True
    with open(syn_texts_tmp, "w", encoding="utf-8") as out_f:
        out_f.write("[")
        for line in _iter_lines(mrconso):
            line_n += 1
            parts = line.rstrip("\n").split("|")
            # Expect >= 17 fields for indices below.
            if len(parts) < 17:
                continue

            cui_int = _cui_to_int(parts[0])
            if cui_int is None or cui_int > max_cui or present[cui_int] == 0:
                continue
            if counts[cui_int] >= max_per:
                continue

            lat = parts[1]
            if lat != "ENG":
                continue

            sab = parts[11]
            if allowed_sabs is not None and sab not in allowed_sabs:
                continue

            tty = parts[12]
            if allowed_ttys is not None and tty not in allowed_ttys:
                continue

            suppress = parts[16]
            if suppress != args.suppress:
                continue

            text = (parts[14] or "").strip()
            if len(text) < int(args.min_len):
                continue

            if first:
                first = False
            else:
                out_f.write(",")
            out_f.write(json.dumps(text, ensure_ascii=False))
            train_cuis.append(int(cui_int))
            counts[cui_int] += 1
            kept += 1

            if line_n % 2_000_000 == 0:
                filled = int((counts >= max_per).sum())
                print(
                    f"  MRCONSO lines={line_n:,} kept={kept:,} "
                    f"filled_cuis={filled:,} (of {int(concept_cuis.shape[0]):,})",
                    flush=True,
                )

        out_f.write("]\n")

    with open(syn_cuis_tmp, "wb") as f:
        np.save(f, np.asarray(train_cuis, dtype=np.int64))
    _atomic_replace(syn_texts_tmp, syn_texts_path)
    _atomic_replace(syn_cuis_tmp, syn_cuis_path)

    filled = int((counts >= max_per).sum())
    print(
        f"[{_now()}] Wrote {syn_texts_path} and {syn_cuis_path}: "
        f"{kept:,} pairs | filled_cuis={filled:,}/{int(concept_cuis.shape[0]):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
