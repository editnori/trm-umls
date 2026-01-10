#!/usr/bin/env python3
"""
Extract a filtered MRREL subset for the current concept universe.

Purpose
  - MRREL contains UMLS concept relations (hierarchy, broader/narrower, etc.).
  - We keep only relations where BOTH CUIs exist in our concept universe
    (`trm_umls/data/embeddings/cui_array.npy`), and only a small set of REL types.

Output (in --output-dir)
  - mrrel_indptr.npy   uint64 [num_concepts + 1]  CSR indptr over concept row indices
  - mrrel_indices.npy  int32  [num_edges]         CSR neighbors (concept row indices)
  - mrrel_metadata.json

Notes
  - This is an offline preprocessing step; MRREL.RRF is large.
  - The CSR indices are based on *concept row indices* (0..num_concepts-1),
    matching `umls_embeddings.npy` / `umls_flat.index` ordering.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import numpy as np


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def _iter_lines(path: Path) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line


def _parse_rel(line: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse only the fields we need from an MRREL line.
    MRREL columns (abridged): CUI1|AUI1|STYPE1|REL|CUI2|...|SUPPRESS|...
    """
    parts = line.rstrip("\n").split("|")
    if len(parts) < 15:
        return None
    return parts[0], parts[3], parts[4], parts[14]


def main() -> None:
    trm_umls_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Extract MRREL subset for the current concept universe (CSR)")
    parser.add_argument("--mrrel", type=Path, required=True, help="Path to MRREL.RRF")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=trm_umls_dir / "data" / "embeddings",
        help="Dir containing cui_array.npy (concept universe ordering)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=trm_umls_dir / "data" / "umls",
        help="Output directory for CSR arrays + metadata",
    )
    parser.add_argument(
        "--keep-rel",
        type=str,
        default="PAR,CHD,RB,RN",
        help="Comma-separated MRREL REL codes to keep (default: hierarchy/broader-narrower)",
    )
    parser.add_argument(
        "--suppress",
        type=str,
        default="N",
        help="SUPPRESS value to keep (default: 'N')",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        default=False,
        help="Store only CUI1->CUI2 edges (default is undirected).",
    )
    args = parser.parse_args()
    undirected = not bool(args.directed)

    if not args.mrrel.exists():
        raise FileNotFoundError(args.mrrel)

    cui_array_path = args.embeddings_dir / "cui_array.npy"
    if not cui_array_path.exists():
        raise FileNotFoundError(f"Missing concept list: {cui_array_path}")

    keep_rel: Set[str] = {r.strip().upper() for r in str(args.keep_rel).split(",") if r.strip()}
    if not keep_rel:
        raise ValueError("--keep-rel must be non-empty (or pass a real list)")

    print(f"[{_now()}] Loading concept universe: {cui_array_path}", flush=True)
    cui_array = np.load(cui_array_path, mmap_mode="r")
    num_concepts = int(cui_array.shape[0])
    max_cui = int(cui_array.max())
    cui_to_row = np.full(max_cui + 1, -1, dtype=np.int32)
    cui_to_row[cui_array.astype(np.int64, copy=False)] = np.arange(num_concepts, dtype=np.int32)
    print(f"[{_now()}] Concepts: {num_concepts:,} | max_cui={max_cui:,}", flush=True)

    # Pass 1: count degrees (CSR sizes)
    print(f"[{_now()}] Pass 1/2: counting degrees from {args.mrrel.name}...", flush=True)
    deg = np.zeros(num_concepts, dtype=np.uint32)
    lines = 0
    kept = 0
    for line in _iter_lines(args.mrrel):
        lines += 1
        parsed = _parse_rel(line)
        if parsed is None:
            continue
        cui1_s, rel_s, cui2_s, suppress_s = parsed
        if suppress_s != args.suppress:
            continue
        rel = (rel_s or "").strip().upper()
        if rel not in keep_rel:
            continue
        cui1 = _cui_to_int(cui1_s)
        cui2 = _cui_to_int(cui2_s)
        if cui1 is None or cui2 is None or cui1 > max_cui or cui2 > max_cui:
            continue
        r1 = int(cui_to_row[cui1])
        r2 = int(cui_to_row[cui2])
        if r1 < 0 or r2 < 0:
            continue

        deg[r1] += 1
        if undirected:
            deg[r2] += 1
        kept += 1

        if lines % 5_000_000 == 0:
            print(f"  lines={lines:,} kept={kept:,}", flush=True)

    indptr = np.zeros(num_concepts + 1, dtype=np.uint64)
    indptr[1:] = np.cumsum(deg, dtype=np.uint64)
    num_edges = int(indptr[-1])
    print(f"[{_now()}] Edges (stored): {num_edges:,} (kept_rel_rows={kept:,})", flush=True)

    # Pass 2: fill CSR indices
    print(f"[{_now()}] Pass 2/2: filling CSR indices...", flush=True)
    indices = np.empty(num_edges, dtype=np.int32)
    cursor = indptr[:-1].copy()
    lines = 0
    kept2 = 0
    for line in _iter_lines(args.mrrel):
        lines += 1
        parsed = _parse_rel(line)
        if parsed is None:
            continue
        cui1_s, rel_s, cui2_s, suppress_s = parsed
        if suppress_s != args.suppress:
            continue
        rel = (rel_s or "").strip().upper()
        if rel not in keep_rel:
            continue
        cui1 = _cui_to_int(cui1_s)
        cui2 = _cui_to_int(cui2_s)
        if cui1 is None or cui2 is None or cui1 > max_cui or cui2 > max_cui:
            continue
        r1 = int(cui_to_row[cui1])
        r2 = int(cui_to_row[cui2])
        if r1 < 0 or r2 < 0:
            continue

        p = int(cursor[r1])
        indices[p] = int(r2)
        cursor[r1] = p + 1
        if undirected:
            p2 = int(cursor[r2])
            indices[p2] = int(r1)
            cursor[r2] = p2 + 1
        kept2 += 1

        if lines % 5_000_000 == 0:
            print(f"  lines={lines:,} kept={kept2:,}", flush=True)

    # Sanity: cursors should match indptr end positions.
    if not np.array_equal(cursor.astype(np.uint64, copy=False), indptr[1:]):
        raise RuntimeError("CSR fill mismatch: cursor != indptr[1:] (bug or corrupted input)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    indptr_path = args.output_dir / "mrrel_indptr.npy"
    indices_path = args.output_dir / "mrrel_indices.npy"
    meta_path = args.output_dir / "mrrel_metadata.json"

    np.save(indptr_path, indptr)
    np.save(indices_path, indices)
    meta_path.write_text(
        json.dumps(
            {
                "created_at": _now(),
                "mrrel_path": str(args.mrrel),
                "num_concepts": num_concepts,
                "max_cui": max_cui,
                "keep_rel": sorted(keep_rel),
                "suppress": args.suppress,
                "undirected": bool(undirected),
                "kept_rel_rows": kept,
                "stored_edges": num_edges,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[{_now()}] Wrote {indptr_path} and {indices_path}", flush=True)
    print(f"[{_now()}] Wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
