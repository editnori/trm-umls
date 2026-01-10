#!/usr/bin/env python3
"""
Teacher bakeoff: compare embedding models for UMLS concept retrieval.

Goal
  Measure how well different embedding "teacher" models map UMLS synonyms to the
  correct CUI when retrieving against a subset of UMLS preferred terms.

Why this exists
  - We don't have gold-labeled clinical mentions.
  - UMLS provides (synonym_text -> CUI) pairs that let us measure retrieval
    quality without manual labeling.

Notes
  - This script is designed to be run on a GPU box (e.g., Lambda A100).
  - It intentionally avoids printing raw text because note data can be sensitive.
  - It uses streaming JSON parsing (ijson) to avoid loading 2.5M strings into RAM.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# FlagEmbedding pulls in `transformers.Trainer` which can trigger optional TF/Flax imports
# (and fail on older system Keras installs). We never need TF/Flax for this script.
# Transformers has changed env var names across versions; set both styles.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _atomic_write_json(path: Path, payload: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _format_secs(seconds: float) -> str:
    seconds = float(max(0.0, seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


def _maybe_set_tf32() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _l2_normalize_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def _reservoir_sample_indices(
    *,
    cuis: np.ndarray,
    allowed_cuis: Set[int],
    sample_size: int,
    seed: int,
) -> List[int]:
    """
    Reservoir sample indices i where cuis[i] in allowed_cuis.
    Returns indices in arbitrary order.
    """
    rng = random.Random(int(seed))
    selected: List[int] = []
    seen = 0
    for i, cui in enumerate(cuis.tolist() if isinstance(cuis, np.ndarray) else cuis):
        if int(cui) not in allowed_cuis:
            continue
        seen += 1
        if len(selected) < sample_size:
            selected.append(int(i))
            continue
        j = rng.randrange(seen)
        if j < sample_size:
            selected[j] = int(i)
    return selected


def _try_import_ijson():
    try:
        import ijson  # type: ignore

        return ijson
    except Exception:
        return None


def _stream_json_array_select(path: Path, wanted_indices_sorted: Sequence[int]) -> List[str]:
    """
    Read a JSON array file and return items at the specified indices.
    Requires indices to be sorted ascending.
    """
    if not wanted_indices_sorted:
        return []

    ijson = _try_import_ijson()
    if ijson is None:
        raise RuntimeError(
            "ijson is required for streaming large JSON arrays. Install with: pip install ijson"
        )

    out: List[str] = []
    ptr = 0
    wanted = wanted_indices_sorted
    target = int(wanted[ptr])

    with open(path, "rb") as f:
        for idx, item in enumerate(ijson.items(f, "item")):
            if idx < target:
                continue
            if idx != target:
                # Should never happen because idx increases by 1.
                continue
            out.append(str(item))
            ptr += 1
            if ptr >= len(wanted):
                break
            target = int(wanted[ptr])

    if len(out) != len(wanted):
        raise RuntimeError(
            f"Failed to stream all items from {path}: got {len(out)}/{len(wanted)}"
        )
    return out


def _stream_embedding_metadata_prefterms(
    embedding_metadata_path: Path, wanted_concept_indices_sorted: Sequence[int]
) -> List[str]:
    """
    Stream `embedding_metadata.json` and select preferred terms by concept index.
    """
    if not wanted_concept_indices_sorted:
        return []

    ijson = _try_import_ijson()
    if ijson is None:
        raise RuntimeError(
            "ijson is required for streaming embedding_metadata.json. Install with: pip install ijson"
        )

    out: List[str] = []
    ptr = 0
    wanted = wanted_concept_indices_sorted
    target = int(wanted[ptr])

    with open(embedding_metadata_path, "rb") as f:
        # Stream items from the `prefterms` array inside the JSON object.
        for idx, term in enumerate(ijson.items(f, "prefterms.item")):
            if idx < target:
                continue
            if idx != target:
                continue
            out.append(str(term))
            ptr += 1
            if ptr >= len(wanted):
                break
            target = int(wanted[ptr])

    if len(out) != len(wanted):
        raise RuntimeError(
            f"Failed to stream all prefterms from {embedding_metadata_path}: got {len(out)}/{len(wanted)}"
        )
    return out


@dataclass
class BakeoffSample:
    concept_indices: List[int]
    concept_cuis: np.ndarray  # int64 [C]
    concept_texts: List[str]  # [C]
    query_indices: List[int]
    query_cuis: np.ndarray  # int64 [Q]
    query_texts: List[str]  # [Q]


def build_sample(
    *,
    base_embeddings_dir: Path,
    seed: int,
    concept_count: int,
    query_count: int,
) -> BakeoffSample:
    """
    Create a deterministic evaluation sample (concept subset + synonym subset).
    """
    cui_array_path = base_embeddings_dir / "cui_array.npy"
    syn_cuis_path = base_embeddings_dir / "synonym_cuis.npy"
    syn_texts_path = base_embeddings_dir / "synonym_texts.json"
    meta_path = base_embeddings_dir / "embedding_metadata.json"

    concept_cuis_all = np.load(cui_array_path, mmap_mode="r")
    num_concepts = int(concept_cuis_all.shape[0])

    rng = np.random.default_rng(int(seed))
    concept_indices = rng.choice(num_concepts, size=int(concept_count), replace=False).astype(int).tolist()
    concept_indices_sorted = sorted(concept_indices)

    concept_cuis = np.asarray(concept_cuis_all[concept_indices_sorted], dtype=np.int64)
    allowed_cuis: Set[int] = set(int(x) for x in concept_cuis.tolist())

    syn_cuis_all = np.load(syn_cuis_path, mmap_mode="r")
    query_indices = _reservoir_sample_indices(
        cuis=syn_cuis_all,
        allowed_cuis=allowed_cuis,
        sample_size=int(query_count),
        seed=int(seed) + 1,
    )
    if not query_indices:
        raise RuntimeError("No synonym queries found for the sampled concept CUIs.")
    query_indices_sorted = sorted(query_indices)
    query_cuis = np.asarray(syn_cuis_all[query_indices_sorted], dtype=np.int64)

    t0 = time.time()
    concept_texts = _stream_embedding_metadata_prefterms(meta_path, concept_indices_sorted)
    query_texts = _stream_json_array_select(syn_texts_path, query_indices_sorted)
    dt = time.time() - t0
    print(f"Loaded sample texts via streaming in {_format_secs(dt)}", flush=True)

    return BakeoffSample(
        concept_indices=concept_indices_sorted,
        concept_cuis=concept_cuis,
        concept_texts=concept_texts,
        query_indices=query_indices_sorted,
        query_cuis=query_cuis,
        query_texts=query_texts,
    )


class TeacherEmbedder:
    name: str
    dim: int

    def embed(self, texts: Sequence[str], *, batch_size: int, device: torch.device) -> np.ndarray:  # noqa: D401
        """Return float32 embeddings [N, dim] normalized."""
        raise NotImplementedError


class PrecomputedNpyEmbedder(TeacherEmbedder):
    def __init__(self, *, name: str, concept_emb_path: Path, synonym_emb_path: Path, dim: int):
        self.name = name
        self.dim = int(dim)
        self._concept = np.load(concept_emb_path, mmap_mode="r")
        self._syn = np.load(synonym_emb_path, mmap_mode="r")

    def embed_concepts(self, indices: Sequence[int]) -> np.ndarray:
        arr = np.asarray(self._concept[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
        return _l2_normalize_np(arr)

    def embed_synonyms(self, indices: Sequence[int]) -> np.ndarray:
        arr = np.asarray(self._syn[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
        return _l2_normalize_np(arr)

    def embed(self, texts: Sequence[str], *, batch_size: int, device: torch.device) -> np.ndarray:
        raise RuntimeError("PrecomputedNpyEmbedder does not embed raw text; use embed_concepts/embed_synonyms.")


class HFTransformerEmbedder(TeacherEmbedder):
    def __init__(self, *, name: str, model_id: str, pooling: str):
        self.name = name
        self.model_id = model_id
        self.pooling = pooling
        self.dim = 0
        self._tok = None
        self._model = None

    def _load(self, device: torch.device):
        if self._tok is not None and self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(device)
        self._model.eval()

        with torch.inference_mode():
            probe = self.embed(["test"], batch_size=1, device=device)
        self.dim = int(probe.shape[1])

    def embed(self, texts: Sequence[str], *, batch_size: int, device: torch.device) -> np.ndarray:
        if self._tok is None or self._model is None:
            self._load(device)
        tok = self._tok
        model = self._model
        assert tok is not None and model is not None

        outs: List[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(texts), int(batch_size)):
                batch = list(texts[i : i + int(batch_size)])
                encoded = tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                out = model(**encoded)
                last = out.last_hidden_state  # [B, T, H]
                if self.pooling == "cls":
                    pooled = last[:, 0]
                elif self.pooling == "mean":
                    mask = encoded.get("attention_mask")
                    if mask is None:
                        pooled = last.mean(dim=1)
                    else:
                        m = mask.unsqueeze(-1).to(last.dtype)
                        pooled = (last * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")

                pooled = F.normalize(pooled, p=2, dim=-1).to(torch.float32).cpu().numpy()
                outs.append(pooled)
        emb = np.concatenate(outs, axis=0) if outs else np.zeros((0, self.dim), dtype=np.float32)
        self.dim = int(emb.shape[1]) if emb.size else int(self.dim)
        return emb


class BGEM3Embedder(TeacherEmbedder):
    def __init__(self, *, name: str, model_id: str):
        self.name = name
        self.model_id = model_id
        self.dim = 0
        self._model = None

    def _load(self, device: torch.device):
        if self._model is not None:
            return
        from FlagEmbedding import BGEM3FlagModel  # type: ignore

        self._model = BGEM3FlagModel(
            self.model_id,
            use_fp16=(device.type == "cuda"),
            device=str(device),
        )
        # Probe dim.
        vec = self.embed(["test"], batch_size=1, device=device)
        self.dim = int(vec.shape[1])

    def embed(self, texts: Sequence[str], *, batch_size: int, device: torch.device) -> np.ndarray:
        if self._model is None:
            self._load(device)
        assert self._model is not None

        # BGEM3 returns a dict with "dense_vecs".
        out = self._model.encode(list(texts), batch_size=int(batch_size), max_length=64)
        dense = out["dense_vecs"]
        emb = np.asarray(dense, dtype=np.float32)
        return _l2_normalize_np(emb)


@torch.inference_mode()
def retrieval_metrics(
    *,
    concept_emb: np.ndarray,
    concept_cuis: np.ndarray,
    query_emb: np.ndarray,
    query_cuis: np.ndarray,
    k: int,
    device: torch.device,
    matmul_batch: int,
) -> Dict[str, float]:
    concept = torch.tensor(concept_emb, device=device, dtype=torch.float16)
    concept = F.normalize(concept, p=2, dim=-1)

    total = int(query_emb.shape[0])
    hits1 = 0
    hitsk = 0
    rr_sum = 0.0

    for i in range(0, total, int(matmul_batch)):
        q = torch.tensor(query_emb[i : i + int(matmul_batch)], device=device, dtype=torch.float16)
        q = F.normalize(q, p=2, dim=-1)
        scores = q @ concept.T  # [B, C]
        top_scores, top_idx = scores.topk(int(k), dim=-1)
        top_idx_cpu = top_idx.to("cpu").numpy()
        # map retrieved indices -> CUIs
        retrieved_cuis = concept_cuis[top_idx_cpu]  # [B, k]
        gold = query_cuis[i : i + int(matmul_batch)].reshape(-1, 1)
        match = (retrieved_cuis == gold)

        hits1 += int(match[:, 0].sum())
        hitsk += int(match.any(axis=1).sum())

        # reciprocal rank (MRR)
        for row in match:
            if not row.any():
                continue
            rank = int(np.argmax(row)) + 1
            rr_sum += 1.0 / float(rank)

    return {
        "queries": float(total),
        "hit@1": hits1 / total,
        f"hit@{k}": hitsk / total,
        "mrr": rr_sum / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher bakeoff for UMLS synonym→CUI retrieval")
    parser.add_argument("--base-embeddings-dir", type=Path, default=Path("trm_umls/data/embeddings"))
    parser.add_argument("--qwen-embeddings-dir", type=Path, default=Path("trm_umls/data/embeddings_qwen3_4096"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concept-count", type=int, default=50000)
    parser.add_argument("--query-count", type=int, default=50000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embed-batch", type=int, default=256)
    parser.add_argument("--matmul-batch", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-sapbert", action="store_true")
    parser.add_argument("--run-bge-m3", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("trm_umls/runs") / f"teacher_bakeoff_{_now_tag()}.json")
    args = parser.parse_args()

    _maybe_set_tf32()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Device: {device}", flush=True)

    print("Building sample...", flush=True)
    t0 = time.time()
    sample = build_sample(
        base_embeddings_dir=Path(args.base_embeddings_dir),
        seed=int(args.seed),
        concept_count=int(args.concept_count),
        query_count=int(args.query_count),
    )
    print(f"Sample: concepts={len(sample.concept_indices):,} queries={len(sample.query_indices):,} in {_format_secs(time.time()-t0)}", flush=True)

    results: Dict[str, Dict[str, float]] = {}
    timings: Dict[str, float] = {}

    # Optional: precomputed "baseline" embeddings (only if both files exist).
    base_concept_path = Path(args.base_embeddings_dir) / "umls_embeddings.npy"
    base_syn_path = Path(args.base_embeddings_dir) / "synonym_embeddings.npy"
    if base_concept_path.exists() and base_syn_path.exists():
        pub = PrecomputedNpyEmbedder(
            name="ModernPubMedBERT(precomputed)",
            concept_emb_path=base_concept_path,
            synonym_emb_path=base_syn_path,
            dim=768,
        )
        print("\n[Teacher] ModernPubMedBERT(precomputed)", flush=True)
        t1 = time.time()
        concept_emb_pub = pub.embed_concepts(sample.concept_indices)
        query_emb_pub = pub.embed_synonyms(sample.query_indices)
        timings["ModernPubMedBERT(precomputed)_embed_s"] = time.time() - t1
        metrics_pub = retrieval_metrics(
            concept_emb=concept_emb_pub,
            concept_cuis=sample.concept_cuis,
            query_emb=query_emb_pub,
            query_cuis=sample.query_cuis,
            k=int(args.k),
            device=device,
            matmul_batch=int(args.matmul_batch),
        )
        results[pub.name] = metrics_pub
        print(metrics_pub, flush=True)
    else:
        missing: List[str] = []
        if not base_concept_path.exists():
            missing.append(str(base_concept_path.name))
        if not base_syn_path.exists():
            missing.append(str(base_syn_path.name))
        print(
            f"\n[Skip] ModernPubMedBERT(precomputed) missing: {', '.join(missing)}",
            flush=True,
        )

    # Baseline: precomputed Qwen3 embeddings (OpenRouter/HF generated).
    q_concept = Path(args.qwen_embeddings_dir) / "umls_embeddings.npy"
    q_syn = Path(args.qwen_embeddings_dir) / "synonym_embeddings.npy"
    if q_concept.exists() and q_syn.exists():
        qwen = PrecomputedNpyEmbedder(
            name="Qwen3-Embedding-8B(precomputed)",
            concept_emb_path=q_concept,
            synonym_emb_path=q_syn,
            dim=4096,
        )
        print("\n[Teacher] Qwen3-Embedding-8B(precomputed)", flush=True)
        t2 = time.time()
        concept_emb_q = qwen.embed_concepts(sample.concept_indices)
        query_emb_q = qwen.embed_synonyms(sample.query_indices)
        timings["Qwen3-Embedding-8B(precomputed)_embed_s"] = time.time() - t2
        metrics_q = retrieval_metrics(
            concept_emb=concept_emb_q,
            concept_cuis=sample.concept_cuis,
            query_emb=query_emb_q,
            query_cuis=sample.query_cuis,
            k=int(args.k),
            device=device,
            matmul_batch=int(args.matmul_batch),
        )
        results[qwen.name] = metrics_q
        print(metrics_q, flush=True)
    else:
        missing: List[str] = []
        if not q_concept.exists():
            missing.append(str(q_concept.name))
        if not q_syn.exists():
            missing.append(str(q_syn.name))
        print(
            f"\n[Skip] Qwen3-Embedding-8B(precomputed) missing: {', '.join(missing)}",
            flush=True,
        )

    # Optional: SapBERT (CLS pooling).
    if args.run_sapbert:
        sap = HFTransformerEmbedder(
            name="SapBERT(cls)",
            model_id="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            pooling="cls",
        )
        print("\n[Teacher] SapBERT(cls)", flush=True)
        t3 = time.time()
        concept_emb_s = sap.embed(sample.concept_texts, batch_size=int(args.embed_batch), device=device)
        query_emb_s = sap.embed(sample.query_texts, batch_size=int(args.embed_batch), device=device)
        timings["SapBERT(cls)_embed_s"] = time.time() - t3
        metrics_s = retrieval_metrics(
            concept_emb=concept_emb_s,
            concept_cuis=sample.concept_cuis,
            query_emb=query_emb_s,
            query_cuis=sample.query_cuis,
            k=int(args.k),
            device=device,
            matmul_batch=int(args.matmul_batch),
        )
        results[sap.name] = metrics_s
        print(metrics_s, flush=True)

    # Optional: BGE-M3 (dense only via FlagEmbedding).
    if args.run_bge_m3:
        bge = BGEM3Embedder(name="BGE-M3(dense)", model_id="BAAI/bge-m3")
        print("\n[Teacher] BGE-M3(dense)", flush=True)
        t4 = time.time()
        concept_emb_b = bge.embed(sample.concept_texts, batch_size=int(args.embed_batch), device=device)
        query_emb_b = bge.embed(sample.query_texts, batch_size=int(args.embed_batch), device=device)
        timings["BGE-M3(dense)_embed_s"] = time.time() - t4
        metrics_b = retrieval_metrics(
            concept_emb=concept_emb_b,
            concept_cuis=sample.concept_cuis,
            query_emb=query_emb_b,
            query_cuis=sample.query_cuis,
            k=int(args.k),
            device=device,
            matmul_batch=int(args.matmul_batch),
        )
        results[bge.name] = metrics_b
        print(metrics_b, flush=True)

    payload = {
        "created_at_unix_s": time.time(),
        "seed": int(args.seed),
        "concept_count": int(len(sample.concept_indices)),
        "query_count": int(len(sample.query_indices)),
        "k": int(args.k),
        "embed_batch": int(args.embed_batch),
        "matmul_batch": int(args.matmul_batch),
        "device": str(device),
        "timings_s": timings,
        "results": results,
    }
    _atomic_write_json(Path(args.output), payload)
    print(f"\nWrote results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
