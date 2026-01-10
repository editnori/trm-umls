#!/usr/bin/env python3
"""
Build FAISS index from UMLS embeddings for fast similarity search.

Supports both exact (IndexFlatIP) and approximate (IndexIVFFlat) search.
"""

import json
import numpy as np
from pathlib import Path
import argparse
import time


def build_faiss_index(
    embeddings_path: Path,
    output_dir: Path,
    index_type: str = "flat",
    nlist: int = 1000,  # for IVF
    nprobe: int = 10,   # for IVF search
    train_sample: int = 200_000,
    seed: int = 13,
):
    """Build FAISS index from embeddings."""
    
    import faiss
    
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path).astype(np.float32)
    print(f"  Shape: {embeddings.shape}")
    
    n_vectors, dim = embeddings.shape
    
    # Normalize embeddings for cosine similarity (inner product on normalized = cosine)
    faiss.normalize_L2(embeddings)
    
    if index_type == "flat":
        # Exact search - slower but exact
        print("Building IndexFlatIP (exact search)...")
        index = faiss.IndexFlatIP(dim)
    elif index_type == "ivf":
        # Approximate search - faster for large datasets
        print(f"Building IndexIVFFlat (approximate, nlist={nlist})...")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print("Training index...")
        train_n = int(train_sample)
        if train_n <= 0 or train_n >= int(n_vectors):
            train_vecs = embeddings
        else:
            rng = np.random.default_rng(int(seed))
            idxs = rng.choice(int(n_vectors), size=train_n, replace=False)
            train_vecs = embeddings[idxs]
        t_train = time.time()
        index.train(train_vecs)
        print(f"  trained on {int(train_vecs.shape[0]):,} vectors in {time.time() - t_train:.1f}s")
        index.nprobe = nprobe
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    print("Adding vectors to index...")
    index.add(embeddings)
    
    print(f"Index contains {index.ntotal:,} vectors")
    
    # Save index
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / f"umls_{index_type}.index"
    faiss.write_index(index, str(index_path))
    print(f"Saved index to {index_path}")
    
    # Test search
    print("\nTesting search...")
    k = 5
    query = embeddings[:3]  # First 3 embeddings as test
    D, I = index.search(query, k)
    print(f"  Top-{k} results for first 3 queries:")
    for i in range(3):
        print(f"    Query {i}: indices={I[i]}, scores={D[i]}")
    
    # Save index config
    config = {
        "index_type": index_type,
        "n_vectors": n_vectors,
        "dim": dim,
        "nlist": nlist if index_type == "ivf" else None,
        "nprobe": nprobe if index_type == "ivf" else None,
        "train_sample": train_sample if index_type == "ivf" else None,
        "seed": seed if index_type == "ivf" else None,
    }
    config_path = output_dir / f"index_config_{index_type}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return index


if __name__ == "__main__":
    trm_umls_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Build FAISS index for UMLS embeddings")
    parser.add_argument(
        "--embeddings", "-e",
        type=Path,
        default=trm_umls_dir / "data" / "embeddings" / "umls_embeddings.npy",
        help="Path to embeddings numpy file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=trm_umls_dir / "data" / "embeddings",
        help="Output directory for FAISS index"
    )
    parser.add_argument(
        "--index-type", "-t",
        type=str,
        choices=["flat", "ivf"],
        default="flat",
        help="Index type: flat (exact) or ivf (approximate)"
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=1000,
        help="Number of clusters for IVF index"
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=10,
        help="Number of clusters to probe at search time (IVF only)",
    )
    parser.add_argument(
        "--train-sample",
        type=int,
        default=200_000,
        help="How many vectors to use for IVF training (0 = use all)",
    )
    parser.add_argument("--seed", type=int, default=13, help="RNG seed for IVF training sample")
    
    args = parser.parse_args()
    
    if not args.embeddings.exists():
        print(f"Error: Embeddings file not found: {args.embeddings}")
        print("Run generate_embeddings_hf_local.py first.")
        exit(1)
    
    build_faiss_index(
        embeddings_path=args.embeddings,
        output_dir=args.output,
        index_type=args.index_type,
        nlist=args.nlist,
        nprobe=int(args.nprobe),
        train_sample=int(args.train_sample),
        seed=int(args.seed),
    )
