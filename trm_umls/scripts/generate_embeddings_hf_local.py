#!/usr/bin/env python3
"""
Generate embeddings locally using a HuggingFace transformer model (GPU).

Primary use case: generate Qwen3 embeddings on an A100 instead of using OpenRouter.

Notes:
- This writes very large files (tens of GB). Make sure you have disk space.
- Output vectors are L2-normalized (cosine similarity via dot product).
- Output format matches the OpenRouter generator so downstream scripts can reuse it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _atomic_write_json(path: Path, payload: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "ETA: ?"
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"ETA: {h}h {m}m {s}s"
    if m:
        return f"ETA: {m}m {s}s"
    return f"ETA: {s}s"


def _estimate_remaining_s(done: int, total: int, elapsed_s: float) -> Optional[float]:
    if done <= 0 or elapsed_s <= 0:
        return None
    rate = done / elapsed_s
    if rate <= 0:
        return None
    return (total - done) / rate


def _pool_mean(last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return last_hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1)
    return summed / denom


@torch.inference_mode()
def embed_batch(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    *,
    device: torch.device,
    max_length: int,
    pooling: str,
    autocast_ctx,
) -> np.ndarray:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with autocast_ctx:
        out = model(**encoded)
        last = out.last_hidden_state
        if pooling == "cls":
            pooled = last[:, 0]
        elif pooling == "mean":
            pooled = _pool_mean(last, encoded.get("attention_mask"))
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        pooled = F.normalize(pooled, p=2, dim=-1)

    return pooled.to(torch.float32).cpu().numpy()


@dataclass
class Phase:
    name: str
    texts: List[str]
    out_path: Path
    limit: Optional[int] = None


def run_phase(
    *,
    phase: Phase,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    embedding_dim: int,
    progress: Dict,
    progress_path: Path,
    resume: bool,
    flush_seconds: float,
    log_seconds: float,
    autocast_ctx,
    pooling: str,
) -> None:
    texts = phase.texts if phase.limit is None else phase.texts[: phase.limit]
    total = len(texts)

    key = f"{phase.name}_offset"
    offset = int(progress.get(key, 0)) if resume else 0
    offset = max(0, min(offset, total))

    if phase.out_path.exists():
        mm = np.load(phase.out_path, mmap_mode="r+")
        if mm.shape != (total, embedding_dim):
            raise ValueError(f"{phase.out_path} has shape {mm.shape}, expected {(total, embedding_dim)}")
        out_mm = mm
    else:
        out_mm = np.lib.format.open_memmap(
            phase.out_path, mode="w+", dtype=np.float32, shape=(total, embedding_dim)
        )

    if offset >= total:
        print(f"[{phase.name}] already complete ({total:,}/{total:,})", flush=True)
        return

    print(
        f"[{phase.name}] start at {offset:,}/{total:,} | batch={batch_size} max_len={max_length} | dim={embedding_dim}",
        flush=True,
    )

    start_t = time.time()
    last_log_t = start_t
    last_flush_t = start_t

    i = offset
    while i < total:
        batch = texts[i : i + batch_size]
        arr = embed_batch(
            model,
            tokenizer,
            batch,
            device=device,
            max_length=max_length,
            pooling=pooling,
            autocast_ctx=autocast_ctx,
        )
        out_mm[i : i + arr.shape[0]] = arr
        i += int(arr.shape[0])

        progress[key] = i
        progress["updated_at_unix_s"] = time.time()
        _atomic_write_json(progress_path, progress)

        now = time.time()
        if now - last_flush_t >= flush_seconds:
            try:
                out_mm.flush()
            except Exception:
                pass
            last_flush_t = now

        if now - last_log_t >= log_seconds:
            elapsed = now - start_t
            done = i - offset
            rate = done / max(1e-9, elapsed)
            eta_s = _estimate_remaining_s(i, total, elapsed)
            print(
                f"[{phase.name}] {i:,}/{total:,} ({100*i/total:.1f}%) | {rate:.1f} texts/s | {_format_eta(eta_s)}",
                flush=True,
            )
            last_log_t = now

    try:
        out_mm.flush()
    except Exception:
        pass
    print(f"[{phase.name}] complete ({total:,}/{total:,})", flush=True)


def main() -> None:
    trm_umls_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Generate embeddings locally with a HuggingFace model")
    parser.add_argument("--input-embeddings-dir", type=Path, default=trm_umls_dir / "data" / "embeddings")
    parser.add_argument("--output-dir", type=Path, default=trm_umls_dir / "data" / "embeddings_qwen3_4096_hf")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--pooling", type=str, choices=["mean", "cls"], default="mean")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-concepts", action="store_true")
    parser.add_argument("--skip-synonyms", action="store_true")
    parser.add_argument("--limit-concepts", type=int, default=None)
    parser.add_argument("--limit-synonyms", type=int, default=None)
    parser.add_argument("--flush-seconds", type=float, default=30.0)
    parser.add_argument("--log-seconds", type=float, default=15.0)
    args = parser.parse_args()

    input_dir = args.input_embeddings_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = input_dir / "embedding_metadata.json"
    syn_texts_path = input_dir / "synonym_texts.json"
    syn_cuis_path = input_dir / "synonym_cuis.npy"
    cui_array_path = input_dir / "cui_array.npy"

    with open(meta_path) as f:
        meta = json.load(f)

    # Preserve original ordering.
    concept_terms: List[str] = meta["prefterms"]
    cuis: List[int] = meta["cuis"]
    if len(concept_terms) != len(cuis):
        raise ValueError(
            "embedding_metadata.json length mismatch: "
            f"len(prefterms)={len(concept_terms):,} vs len(cuis)={len(cuis):,}"
        )

    # Ensure concept universe file exists in the output dir.
    out_cui_array = output_dir / "cui_array.npy"
    if not out_cui_array.exists():
        if cui_array_path.exists():
            shutil.copyfile(cui_array_path, out_cui_array)
        else:
            np.save(out_cui_array, np.asarray(cuis, dtype=np.int64))

    # Copy synonym metadata so training can point at the output dir, without
    # forcing us to load a huge JSON list unless we actually embed synonyms.
    synonym_texts: List[str] = []
    if syn_texts_path.exists() and not (output_dir / "synonym_texts.json").exists():
        shutil.copyfile(syn_texts_path, output_dir / "synonym_texts.json")
    if syn_cuis_path.exists() and not (output_dir / "synonym_cuis.npy").exists():
        shutil.copyfile(syn_cuis_path, output_dir / "synonym_cuis.npy")

    if not args.skip_synonyms:
        if not syn_texts_path.exists() or not syn_cuis_path.exists():
            raise FileNotFoundError(
                f"Missing synonym files in {input_dir}: {syn_texts_path.name} and/or {syn_cuis_path.name}"
            )
        with open(syn_texts_path) as f:
            synonym_texts = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Input embeddings dir: {input_dir}")
    print(f"Output embeddings dir: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Device: {device} dtype={args.dtype} batch={args.batch_size} max_length={args.max_length}")
    print(f"Pooling: {args.pooling}")

    print("Loading tokenizer/model...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).to(device)
    model.eval()

    autocast_ctx = nullcontext()
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype)

    # Probe embedding dimension once.
    probe = embed_batch(
        model,
        tok,
        ["test"],
        device=device,
        max_length=int(args.max_length),
        pooling=str(args.pooling),
        autocast_ctx=autocast_ctx,
    )
    embedding_dim = int(probe.shape[1])
    print(f"Embedding dim: {embedding_dim}", flush=True)

    # Write updated metadata in the output dir (reuse cuis/prefterms order).
    out_meta = dict(meta)
    out_meta["model"] = args.model
    out_meta["embedding_dim"] = embedding_dim
    out_meta["generated_by"] = "hf_local"
    out_meta["generated_at_unix_s"] = time.time()
    with open(output_dir / "embedding_metadata.json", "w") as f:
        json.dump(out_meta, f)

    progress_path = output_dir / "hf_progress.json"
    progress: Dict = {}
    if args.resume and progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)

    progress.setdefault("model", args.model)
    progress.setdefault("embedding_dim", embedding_dim)
    progress.setdefault("batch_size", int(args.batch_size))
    progress.setdefault("max_length", int(args.max_length))
    progress.setdefault("dtype", args.dtype)
    progress.setdefault("created_at_unix_s", time.time())
    _atomic_write_json(progress_path, progress)

    phases: List[Phase] = []
    if not args.skip_concepts:
        phases.append(
            Phase(
                name="concepts",
                texts=concept_terms,
                out_path=output_dir / "umls_embeddings.npy",
                limit=args.limit_concepts,
            )
        )
    if not args.skip_synonyms:
        phases.append(
            Phase(
                name="synonyms",
                texts=synonym_texts,
                out_path=output_dir / "synonym_embeddings.npy",
                limit=args.limit_synonyms,
            )
        )

    for phase in phases:
        run_phase(
            phase=phase,
            model=model,
            tokenizer=tok,
            device=device,
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            embedding_dim=embedding_dim,
            progress=progress,
            progress_path=progress_path,
            resume=bool(args.resume),
            flush_seconds=float(args.flush_seconds),
            log_seconds=float(args.log_seconds),
            autocast_ctx=autocast_ctx,
            pooling=str(args.pooling),
        )

    print("\nAll requested phases complete.", flush=True)


if __name__ == "__main__":
    main()
