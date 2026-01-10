#!/usr/bin/env python3
"""
Link span mentions to CUIs using a SapBERT-style embedding + FAISS index.

Input: JSONL from biomedical NER teacher (mentions with text/start/end)
Output: JSONL with span -> CUI predictions (silver labels)

FLOW: link_spans_to_cuis
Entrypoint: `python trm_umls/scripts/link_spans_to_cuis.py ...`
Inputs:
  - `--labels` one or more JSONL files from `teacher_label_notes_*.py`
  - `--embeddings-dir` containing: `embedding_metadata.json` + `umls_flat.index`
Happy path:
  1) Read teacher mentions and batch span texts.
  2) Embed spans in the teacher space (SapBERT by default).
  3) FAISS top-k retrieve CUIs for each span.
  4) Optional lexical rerank within a small similarity margin.
  5) Apply min_sim/min_lex filters and write JSONL rows (silver labels).
Outputs:
  - JSONL at `--output-jsonl` (one row per kept mention)
Observability:
  - Periodic `processed=... kept=...` logs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# Transformers sometimes tries to import TF/Flax; we only need PyTorch.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")


def _load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_label_paths(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".jsonl"]))
        else:
            out.append(p)
    return out


def _token_jaccard(a: str, b: str) -> float:
    def _tok(s: str) -> set[str]:
        return {t for t in str(s).lower().replace("/", " ").split() if t}
    sa = _tok(a)
    sb = _tok(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union) if union else 0.0


@torch.inference_mode()
def _embed_batch(model, tokenizer, texts: List[str], *, device: torch.device, max_length: int, pooling: str) -> np.ndarray:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    out = model(**encoded)
    last = out.last_hidden_state
    if pooling == "cls":
        pooled = last[:, 0]
    else:
        mask = encoded.get("attention_mask")
        if mask is None:
            pooled = last.mean(dim=1)
        else:
            mask = mask.unsqueeze(-1).to(last.dtype)
            summed = (last * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            pooled = summed / denom
    pooled = F.normalize(pooled, p=2, dim=-1)
    return pooled.to(torch.float32).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Link NER spans to CUIs via SapBERT + FAISS")
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model", type=str, default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument("--pooling", type=str, choices=["cls", "mean"], default="cls")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-rerank", action="store_true", help="Disable lexical rerank within the top-k")
    parser.add_argument("--lexical-weight", type=float, default=0.30, help="Rerank weight for token overlap")
    parser.add_argument(
        "--rerank-margin",
        type=float,
        default=0.04,
        help="Only allow rerank to switch within this similarity delta of the top FAISS hit (set <0 to disable).",
    )
    parser.add_argument("--min-sim", type=float, default=0.60)
    parser.add_argument("--min-lex", type=float, default=0.10)
    parser.add_argument("--min-ner-score", type=float, default=0.20)
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument("--limit-mentions", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=10_000)
    args = parser.parse_args()

    label_paths = _iter_label_paths(args.labels)
    if not label_paths:
        raise SystemExit("No label JSONL files found via --labels")

    emb_dir = args.embeddings_dir
    meta_path = emb_dir / "embedding_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    if args.index_path is None:
        index_path = emb_dir / "umls_flat.index"
    else:
        index_path = args.index_path
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    cuis = np.asarray(meta["cuis"], dtype=np.int64)
    prefterms: List[str] = list(meta["prefterms"])

    import faiss

    index = faiss.read_index(str(index_path))
    if index.ntotal != len(prefterms):
        raise ValueError("Index/metadata size mismatch")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype="auto").to(device)
    model.eval()

    out_path = args.output_jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)

    min_chars = int(args.min_chars)
    max_chars = int(args.max_chars)
    min_sim = float(args.min_sim)
    min_lex = float(args.min_lex)
    min_ner = float(args.min_ner_score)
    do_rerank = not bool(args.no_rerank)
    lex_w = float(args.lexical_weight)
    margin = float(args.rerank_margin)
    log_every = max(1, int(args.log_every))

    batch_texts: List[str] = []
    batch_meta: List[Dict[str, Any]] = []

    total_mentions = 0
    kept = 0
    t0 = time.time()

    def flush_batch(out_f):
        nonlocal kept
        if not batch_texts:
            return
        emb = _embed_batch(model, tok, batch_texts, device=device, max_length=int(args.max_length), pooling=str(args.pooling))
        D, I = index.search(emb, int(args.top_k))

        for i, meta_row in enumerate(batch_meta):
            sims = D[i]
            idxs = I[i]
            lex_text = str(meta_row.get("text_for_embed") or meta_row.get("normalized") or meta_row.get("text") or "")

            # Default: top FAISS hit.
            top_sim = float(sims[0])
            best_j = 0
            best_sim = top_sim
            best_idx = int(idxs[0])
            best_pref = prefterms[best_idx]
            best_lex = _token_jaccard(lex_text, best_pref)
            best_score = best_sim + (lex_w * best_lex if do_rerank else 0.0)

            # Optional rerank: allow lexical match to win when it is close in embedding space.
            if do_rerank and int(args.top_k) > 1:
                for j in range(1, int(args.top_k)):
                    sim_j = float(sims[j])
                    if margin >= 0 and sim_j < top_sim - margin:
                        # Scores are sorted by sim; if we're outside the margin, stop early.
                        break
                    idx_j = int(idxs[j])
                    pref_j = prefterms[idx_j]
                    lex_j = _token_jaccard(lex_text, pref_j)
                    score_j = sim_j + (lex_w * lex_j)
                    if score_j > best_score:
                        best_score = score_j
                        best_j = j
                        best_sim = sim_j
                        best_idx = idx_j
                        best_pref = pref_j
                        best_lex = lex_j

            # Optional: require minimal similarity and lexical overlap.
            if best_sim < min_sim or (min_lex > 0 and best_lex < min_lex):
                continue

            row = {
                "note_file": meta_row["note_file"],
                "note_type": meta_row["note_type"],
                "start": meta_row["start"],
                "end": meta_row["end"],
                "text": meta_row["text"],
                "normalized": meta_row.get("normalized") or "",
                "label": meta_row["label"],
                "ner_score": meta_row["ner_score"],
                "teacher_count": int(meta_row.get("teacher_count") or 0),
                "teachers": meta_row.get("teachers") or [],
                "teacher_max_score": meta_row.get("teacher_max_score"),
                "teacher_mean_score": meta_row.get("teacher_mean_score"),
                "cui": f"C{int(cuis[best_idx]):07d}",
                "preferred_term": str(best_pref),
                "sim": float(best_sim),
                "lex": float(best_lex),
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

        batch_texts.clear()
        batch_meta.clear()

    with out_path.open("w", encoding="utf-8") as out_f:
        for lp in label_paths:
            for row in _load_jsonl(lp):
                note_file = str(row.get("note_file") or "")
                note_type = str(row.get("note_type") or "")
                for m in (row.get("mentions") or []):
                    raw_text = str(m.get("text") or m.get("mention") or "").strip()
                    normalized = str(m.get("normalized") or "").strip()
                    text_for_embed = normalized or raw_text
                    if not text_for_embed:
                        continue
                    if len(text_for_embed) < min_chars or len(text_for_embed) > max_chars:
                        continue
                    try:
                        ner_score = float(m.get("score") or 0.0)
                    except Exception:
                        ner_score = 0.0
                    if ner_score < min_ner:
                        continue
                    start = int(m.get("start", -1))
                    end = int(m.get("end", -1))
                    label = str(m.get("label") or "")
                    teacher_count = int(m.get("teacher_count", 0) or 0)
                    teachers = m.get("teachers") or []
                    max_score = m.get("max_score")
                    mean_score = m.get("mean_score")

                    batch_texts.append(text_for_embed)
                    batch_meta.append(
                        {
                            "note_file": note_file,
                            "note_type": note_type,
                            "start": start,
                            "end": end,
                            "label": label,
                            "ner_score": ner_score,
                            "text": raw_text,
                            "normalized": normalized,
                            "text_for_embed": text_for_embed,
                            "teacher_count": teacher_count,
                            "teachers": teachers,
                            "teacher_max_score": max_score,
                            "teacher_mean_score": mean_score,
                        }
                    )
                    total_mentions += 1

                    if args.limit_mentions is not None and total_mentions >= int(args.limit_mentions):
                        flush_batch(out_f)
                        elapsed = time.time() - t0
                        print(f"Stopped at limit_mentions={args.limit_mentions} kept={kept} elapsed_s={elapsed:.1f}")
                        return

                    if len(batch_texts) >= int(args.batch_size):
                        flush_batch(out_f)

                    if total_mentions % log_every == 0:
                        elapsed = time.time() - t0
                        rate = total_mentions / max(1e-9, elapsed)
                        print(
                            f"processed={total_mentions:,} kept={kept:,} ({100.0*kept/max(1,total_mentions):.1f}%) "
                            f"rate={rate:.1f} mentions/s elapsed_s={elapsed:.1f}",
                            flush=True,
                        )

        flush_batch(out_f)

    elapsed = time.time() - t0
    print(f"DONE mentions={total_mentions} kept={kept} elapsed_s={elapsed:.1f}")


if __name__ == "__main__":
    main()
