#!/usr/bin/env python3
"""
Multi-task training for TRM-UMLS.

Trains the model with three objectives:
1. Embedding: Match a teacher embedding space (contrastive loss)
2. Assertion: Classify PRESENT/ABSENT/POSSIBLE
3. Subject: Classify PATIENT/FAMILY/OTHER

TUI/Semantic Type is obtained from CUI→TUI lookup after matching.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

try:
    from .models.trm_text_encoder import TRMTextEncoder, TRMTextEncoderConfig
except ImportError:  # pragma: no cover
    from models.trm_text_encoder import TRMTextEncoder, TRMTextEncoderConfig


# Label mappings
ASSERTION_LABELS = {"PRESENT": 0, "ABSENT": 1, "POSSIBLE": 2}
SUBJECT_LABELS = {"PATIENT": 0, "FAMILY": 1, "OTHER": 2}


class MultiTaskDataset(Dataset):
    """Dataset that combines embedding, assertion, and subject data."""
    
    def __init__(
        self,
        embedding_data: Dict,       # synonym_texts + targets (synonym or concept)
        assertion_data: List[Dict], # text, label
        subject_data: List[Dict],   # text, label
        tokenizer,
        max_length: int = 64,
        max_samples: int = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Embedding task data
        self.embedding_texts = embedding_data["texts"]
        self.target_mode = str(embedding_data.get("target_mode", "synonym"))
        if self.target_mode == "synonym":
            self.embedding_targets = embedding_data["embeddings"]
            self.concept_embeddings = None
            self.concept_rows = None
        elif self.target_mode == "concept":
            # Targets are concept (prefterm) embeddings indexed by a row per synonym.
            self.embedding_targets = None
            self.concept_embeddings = embedding_data["concept_embeddings"]
            self.concept_rows = embedding_data["concept_rows"]
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode!r}")

        if max_samples and len(self.embedding_texts) > max_samples:
            indices = random.sample(range(len(self.embedding_texts)), int(max_samples))
            self.embedding_texts = [self.embedding_texts[i] for i in indices]
            if self.target_mode == "synonym":
                self.embedding_targets = self.embedding_targets[indices]
            else:
                self.concept_rows = self.concept_rows[indices]
        
        # Assertion task data
        self.assertion_data = assertion_data
        
        # Subject task data
        self.subject_data = subject_data
        
        # Determine dataset length (use embedding data as primary)
        self.length = len(self.embedding_texts)
        
        print(f"Dataset initialized:")
        print(f"  Embedding samples: {len(self.embedding_texts)} (target_mode={self.target_mode})")
        print(f"  Assertion samples: {len(self.assertion_data)}")
        print(f"  Subject samples: {len(self.subject_data)}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Embedding task
        emb_text = self.embedding_texts[idx]
        emb_concept_row = -1
        if self.target_mode == "synonym":
            emb_target = self.embedding_targets[idx]
        else:
            row = int(self.concept_rows[idx])
            emb_concept_row = row
            emb_target = self.concept_embeddings[row]

        # Assertion task (sample from assertion data)
        assert_idx = idx % len(self.assertion_data)
        assert_item = self.assertion_data[assert_idx]
        assert_text = assert_item["text"]
        assert_label = ASSERTION_LABELS.get(assert_item["label"], 0)

        # Subject task (sample from subject data)
        subj_idx = idx % len(self.subject_data)
        subj_item = self.subject_data[subj_idx]
        subj_text = subj_item["text"]
        subj_label = SUBJECT_LABELS.get(subj_item["label"], 0)

        return {
            # Raw texts; tokenized in the dataloader collate_fn for speed.
            "emb_text": emb_text,
            "emb_target": emb_target,
            "emb_concept_row": emb_concept_row,
            "assert_text": assert_text,
            "assert_label": assert_label,
            "subj_text": subj_text,
            "subj_label": subj_label,
        }

    def collate_fn(self, items: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Batch tokenization collate function.

        This is much faster than calling the tokenizer in `__getitem__()`.
        """
        emb_texts = [it["emb_text"] for it in items]
        assert_texts = [it["assert_text"] for it in items]
        subj_texts = [it["subj_text"] for it in items]

        emb_encoded = self.tokenizer(
            emb_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        assert_encoded = self.tokenizer(
            assert_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        subj_encoded = self.tokenizer(
            subj_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        emb_targets = np.stack([it["emb_target"] for it in items]).astype(np.float32, copy=False)
        emb_concept_rows = np.asarray([it["emb_concept_row"] for it in items], dtype=np.int64)
        assert_labels = np.asarray([it["assert_label"] for it in items], dtype=np.int64)
        subj_labels = np.asarray([it["subj_label"] for it in items], dtype=np.int64)

        return {
            "emb_input_ids": emb_encoded["input_ids"],
            "emb_attention_mask": emb_encoded["attention_mask"],
            "emb_target": torch.from_numpy(emb_targets),
            "emb_concept_row": torch.from_numpy(emb_concept_rows),
            "assert_input_ids": assert_encoded["input_ids"],
            "assert_attention_mask": assert_encoded["attention_mask"],
            "assert_label": torch.from_numpy(assert_labels),
            "subj_input_ids": subj_encoded["input_ids"],
            "subj_attention_mask": subj_encoded["attention_mask"],
            "subj_label": torch.from_numpy(subj_labels),
        }


def contrastive_loss(pred_emb: torch.Tensor, target_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE contrastive loss."""
    pred_emb = F.normalize(pred_emb, p=2, dim=-1)
    target_emb = F.normalize(target_emb, p=2, dim=-1)
    
    logits = torch.matmul(pred_emb, target_emb.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    return F.cross_entropy(logits, labels)


def hard_negative_loss(
    pred_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Hard-negative loss: for each query embedding, classify the positive embedding among hard negatives.

    Shapes:
      pred_emb: [B, D]
      pos_emb:  [B, D]
      neg_emb:  [B, K, D]
    """
    pred_emb = F.normalize(pred_emb, p=2, dim=-1)
    pos_emb = F.normalize(pos_emb, p=2, dim=-1)
    neg_emb = F.normalize(neg_emb, p=2, dim=-1)

    pos_logits = (pred_emb * pos_emb).sum(dim=-1, keepdim=True)  # [B, 1]
    neg_logits = (pred_emb.unsqueeze(1) * neg_emb).sum(dim=-1)  # [B, K]
    logits = torch.cat([pos_logits, neg_logits], dim=1) / float(temperature)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def train_epoch(
    model: TRMTextEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    loss_weights: Dict[str, float],
    concept_embeddings=None,
    concept_neighbors=None,
    hard_neg_k: int = 0,
    hard_neg_mix: float = 0.0,
    temperature: float = 0.07,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    emb_loss_sum = 0.0
    hard_loss_sum = 0.0
    assert_loss_sum = 0.0
    subj_loss_sum = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        optimizer.zero_grad()
        
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # Embedding task
            emb_outputs = model(
                batch["emb_input_ids"].to(device),
                batch["emb_attention_mask"].to(device),
                task="embedding"
            )
            emb_target = batch["emb_target"].to(device)
            base_emb_loss = contrastive_loss(emb_outputs["embedding"], emb_target, temperature=float(temperature))

            # Optional: concept-centered hard negatives (requires concept targets).
            emb_loss = base_emb_loss
            hard_loss = None
            if (
                concept_embeddings is not None
                and concept_neighbors is not None
                and hard_neg_k > 0
                and hard_neg_mix > 0.0
            ):
                concept_rows = batch.get("emb_concept_row")
                if concept_rows is not None:
                    rows_np = concept_rows.cpu().numpy().astype(np.int64, copy=False)
                    neg_rows = concept_neighbors[rows_np, : int(hard_neg_k)]
                    if (neg_rows < 0).any():
                        fill = (rows_np[:, None] + 1) % int(concept_neighbors.shape[0])
                        neg_rows = np.where(neg_rows < 0, fill, neg_rows)
                    neg_emb_np = concept_embeddings[neg_rows]
                    neg_emb = torch.from_numpy(neg_emb_np).to(device=device, dtype=torch.float32, non_blocking=True)
                    hard_loss = hard_negative_loss(
                        emb_outputs["embedding"],
                        emb_target,
                        neg_emb,
                        temperature=float(temperature),
                    )
                    emb_loss = (1.0 - float(hard_neg_mix)) * base_emb_loss + float(hard_neg_mix) * hard_loss
            
            # Assertion task
            assert_outputs = model(
                batch["assert_input_ids"].to(device),
                batch["assert_attention_mask"].to(device),
                task="assertion"
            )
            assert_labels = batch["assert_label"].to(device)
            assert_loss = F.cross_entropy(assert_outputs["assertion_logits"], assert_labels)
            
            # Subject task
            subj_outputs = model(
                batch["subj_input_ids"].to(device),
                batch["subj_attention_mask"].to(device),
                task="subject"
            )
            subj_labels = batch["subj_label"].to(device)
            subj_loss = F.cross_entropy(subj_outputs["subject_logits"], subj_labels)
            
            # Combined loss
            loss = (
                loss_weights["embedding"] * emb_loss +
                loss_weights["assertion"] * assert_loss +
                loss_weights["subject"] * subj_loss
            )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        emb_loss_sum += emb_loss.item()
        if hard_loss is not None:
            hard_loss_sum += hard_loss.item()
        assert_loss_sum += assert_loss.item()
        subj_loss_sum += subj_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "emb": f"{emb_loss.item():.4f}",
            "hard": f"{(hard_loss.item() if hard_loss is not None else 0.0):.4f}",
            "assert": f"{assert_loss.item():.4f}",
            "subj": f"{subj_loss.item():.4f}"
        })
    
    return {
        "total_loss": total_loss / num_batches,
        "emb_loss": emb_loss_sum / num_batches,
        "hard_emb_loss": hard_loss_sum / max(1, num_batches),
        "assert_loss": assert_loss_sum / num_batches,
        "subj_loss": subj_loss_sum / num_batches,
    }


def validate(
    model: TRMTextEncoder,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    emb_sims = []
    assert_correct = 0
    assert_total = 0
    subj_correct = 0
    subj_total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Embedding task - compute similarity
            emb_outputs = model(
                batch["emb_input_ids"].to(device),
                batch["emb_attention_mask"].to(device),
                task="embedding"
            )
            emb_target = batch["emb_target"].to(device)
            emb_target = F.normalize(emb_target, p=2, dim=-1)
            
            cos_sim = (emb_outputs["embedding"] * emb_target).sum(dim=-1)
            emb_sims.extend(cos_sim.cpu().tolist())
            
            # Assertion task - accuracy
            assert_outputs = model(
                batch["assert_input_ids"].to(device),
                batch["assert_attention_mask"].to(device),
                task="assertion"
            )
            assert_preds = assert_outputs["assertion_logits"].argmax(dim=-1)
            assert_labels = batch["assert_label"].to(device)
            assert_correct += (assert_preds == assert_labels).sum().item()
            assert_total += len(assert_labels)
            
            # Subject task - accuracy
            subj_outputs = model(
                batch["subj_input_ids"].to(device),
                batch["subj_attention_mask"].to(device),
                task="subject"
            )
            subj_preds = subj_outputs["subject_logits"].argmax(dim=-1)
            subj_labels = batch["subj_label"].to(device)
            subj_correct += (subj_preds == subj_labels).sum().item()
            subj_total += len(subj_labels)
    
    return {
        "emb_mean_sim": np.mean(emb_sims),
        "emb_median_sim": np.median(emb_sims),
        "assert_accuracy": assert_correct / assert_total if assert_total > 0 else 0,
        "subj_accuracy": subj_correct / subj_total if subj_total > 0 else 0,
    }


def main(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if bool(args.resume) and args.init_checkpoint is not None:
        raise SystemExit("Choose only one: --resume or --init-checkpoint")

    resume_state = None
    resume_path = None
    if args.resume:
        if args.resume_checkpoint is not None:
            resume_path = args.resume_checkpoint
        else:
            model_last = checkpoint_dir / "model_last.pt"
            model_best = checkpoint_dir / "model.pt"
            resume_path = model_last if model_last.exists() else model_best

        if not resume_path.exists():
            raise FileNotFoundError(
                f"--resume was set but checkpoint not found: {resume_path}"
            )
        print(f"Resuming from: {resume_path}")
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)

    init_state = None
    init_path = None
    if args.init_checkpoint is not None:
        init_path = args.init_checkpoint
        if not init_path.exists():
            raise FileNotFoundError(init_path)
        print(f"Initializing weights from: {init_path} (model only; epoch/optimizer reset)")
        init_state = torch.load(init_path, map_location="cpu", weights_only=False)
    
    # Load data
    print("\nLoading data...")
    
    # Embedding data
    emb_dir = Path(args.embedding_dir)
    
    # Load synonym texts (stored as JSON)
    with open(emb_dir / "synonym_texts.json") as f:
        synonym_texts = json.load(f)

    target_mode = str(args.embedding_target).strip().lower()
    if target_mode not in ("synonym", "concept"):
        raise ValueError("--embedding-target must be 'synonym' or 'concept'")

    print(f"  Embedding texts: {len(synonym_texts)} (target_mode={target_mode})")

    concept_embeddings = None
    if target_mode == "synonym":
        synonym_embeddings = np.load(emb_dir / "synonym_embeddings.npy", mmap_mode="r")
        teacher_embedding_dim = int(synonym_embeddings.shape[1])
        print(f"  Target vectors: {synonym_embeddings.shape} (synonym_embeddings.npy)")
        embedding_data = {
            "texts": synonym_texts,
            "target_mode": "synonym",
            "embeddings": synonym_embeddings,
        }
    else:
        concept_embeddings = np.load(emb_dir / "umls_embeddings.npy", mmap_mode="r")
        teacher_embedding_dim = int(concept_embeddings.shape[1])

        cui_array_path = emb_dir / "cui_array.npy"
        if not cui_array_path.exists():
            raise FileNotFoundError(f"Missing {cui_array_path} (required for concept targets)")
        cui_array = np.load(cui_array_path, mmap_mode="r")
        if int(cui_array.shape[0]) != int(concept_embeddings.shape[0]):
            raise ValueError(
                "umls_embeddings.npy and cui_array.npy length mismatch: "
                f"{int(concept_embeddings.shape[0])} vs {int(cui_array.shape[0])}"
            )

        syn_cuis_path = emb_dir / "synonym_cuis.npy"
        if not syn_cuis_path.exists():
            raise FileNotFoundError(f"Missing {syn_cuis_path} (required for concept targets)")
        synonym_cuis = np.load(syn_cuis_path, mmap_mode="r")

        # Build a dense CUI->row mapping for fast vectorized indexing.
        max_cui = int(cui_array.max())
        cui_to_row = np.full(max_cui + 1, -1, dtype=np.int32)
        cui_to_row[cui_array.astype(np.int64, copy=False)] = np.arange(int(cui_array.shape[0]), dtype=np.int32)

        if int(synonym_cuis.max()) > max_cui:
            raise ValueError(
                "synonym_cuis contains CUIs outside cui_array range: "
                f"max_syn_cui={int(synonym_cuis.max())} > max_cui={max_cui}"
            )

        concept_rows = cui_to_row[synonym_cuis.astype(np.int64, copy=False)]
        missing = int((concept_rows < 0).sum())
        if missing:
            raise ValueError(
                f"Failed to map {missing:,} synonym CUIs to concept rows. "
                "Rebuild embeddings so the concept universe matches training pairs."
            )

        print(f"  Target vectors: {concept_embeddings.shape} (umls_embeddings.npy)")
        print(f"  Training pairs: {int(concept_rows.shape[0]):,} (synonym_texts.json + synonym_cuis.npy)")

        embedding_data = {
            "texts": synonym_texts,
            "target_mode": "concept",
            "concept_embeddings": concept_embeddings,
            "concept_rows": concept_rows,
        }

    # Optional: hard-negative neighbors (concept-mode only)
    concept_neighbors = None
    hard_neg_k = 0
    hard_neg_mix = 0.0
    if args.hard_negs is not None and float(args.hard_neg_mix) > 0.0:
        if target_mode != "concept":
            raise ValueError("--hard-negs/--hard-neg-mix requires --embedding-target concept")
        hard_path = Path(args.hard_negs)
        if not hard_path.exists():
            raise FileNotFoundError(hard_path)
        concept_neighbors = np.load(hard_path, mmap_mode="r")
        if int(concept_neighbors.shape[0]) != int(concept_embeddings.shape[0]):
            raise ValueError(
                "Hard-neg neighbors must match concept universe size: "
                f"neighbors={int(concept_neighbors.shape[0])} vs concepts={int(concept_embeddings.shape[0])}"
            )
        hard_neg_k = max(0, int(args.hard_neg_k))
        if hard_neg_k and int(concept_neighbors.shape[1]) < hard_neg_k:
            raise ValueError(
                f"--hard-neg-k={hard_neg_k} but neighbors only has {int(concept_neighbors.shape[1])} columns"
            )
        hard_neg_mix = float(args.hard_neg_mix)
        print(f"  Hard negatives: {hard_path} | k={hard_neg_k} | mix={hard_neg_mix}")
    
    # Assertion data
    with open(args.assertion_data) as f:
        assertion_data = json.load(f)
    
    # Subject data
    with open(args.subject_data) as f:
        subject_data = json.load(f)
    
    # Create dataset
    dataset = MultiTaskDataset(
        embedding_data=embedding_data,
        assertion_data=assertion_data,
        subject_data=subject_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_samples
    )
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    
    # Create model
    print("\nCreating model...")

    best_val_sim = 0.0
    best_epoch = 0
    start_epoch = 1

    if resume_state is not None:
        config = TRMTextEncoderConfig(**resume_state["config"])
        if int(config.embedding_dim) != teacher_embedding_dim:
            raise ValueError(
                "Embedding dim mismatch between checkpoint and embedding-dir: "
                f"checkpoint embedding_dim={int(config.embedding_dim)}, "
                f"teacher embedding_dim={teacher_embedding_dim}. "
                "Use a checkpoint trained for this embedding-dir or pass --embedding-dir that matches the checkpoint."
            )
        model = TRMTextEncoder(config)
        model.load_state_dict(resume_state["model_state_dict"])
        best_val_sim = float(resume_state.get("best_val_sim", 0.0))
        best_epoch = int(resume_state.get("best_epoch", 0))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        if args.v1_checkpoint:
            print("Note: ignoring --v1-checkpoint because --resume is set.")
        if args.init_checkpoint is not None:
            print("Note: ignoring --init-checkpoint because --resume is set.")
    elif init_state is not None:
        config = TRMTextEncoderConfig(**init_state["config"])
        if int(config.embedding_dim) != teacher_embedding_dim:
            raise ValueError(
                "Embedding dim mismatch between init-checkpoint and embedding-dir: "
                f"init embedding_dim={int(config.embedding_dim)}, teacher embedding_dim={teacher_embedding_dim}. "
                "Use a checkpoint trained for this embedding-dir or pass a matching --embedding-dir."
            )
        model = TRMTextEncoder(config)
        model.load_state_dict(init_state["model_state_dict"])
        if args.v1_checkpoint:
            print("Note: ignoring --v1-checkpoint because --init-checkpoint is set.")
    else:
        config = TRMTextEncoderConfig(
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            h_cycles=args.h_cycles,
            l_cycles=args.l_cycles,
            embedding_dim=teacher_embedding_dim,
            vocab_size=tokenizer.vocab_size,
            max_seq_len=args.max_length,
            num_assertion_labels=3,
            num_subject_labels=3,
            classifier_hidden_size=args.classifier_hidden_size,
            classifier_dropout=args.classifier_dropout,
        )

        # Load from V1 checkpoint if provided
        if args.v1_checkpoint:
            print(f"Loading from V1 checkpoint: {args.v1_checkpoint}")
            model = TRMTextEncoder.from_pretrained_v1(Path(args.v1_checkpoint), config)
        else:
            model = TRMTextEncoder(config)
    
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
    
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    
    # Loss weights
    loss_weights = {
        "embedding": args.emb_weight,
        "assertion": args.assert_weight,
        "subject": args.subj_weight,
    }
    print(f"Loss weights: {loss_weights}")
    
    # Training loop
    if start_epoch > args.epochs:
        print(
            f"Nothing to do: resume epoch {start_epoch} > --epochs {args.epochs}. "
            "Increase --epochs to continue training."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            loss_weights,
            concept_embeddings=concept_embeddings if concept_neighbors is not None else None,
            concept_neighbors=concept_neighbors,
            hard_neg_k=hard_neg_k,
            hard_neg_mix=hard_neg_mix,
            temperature=float(args.temperature),
        )
        
        print(f"\nTrain metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        print(f"\nValidation metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.model_dump(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_epoch": best_epoch,
            "best_val_sim": best_val_sim,
            "teacher_target_mode": target_mode,
            "init_from_checkpoint": str(init_path) if init_path is not None else None,
        }

        # Always save a resumable checkpoint.
        last_path = checkpoint_dir / "model_last.pt"
        last_tmp = checkpoint_dir / "model_last.pt.tmp"
        torch.save(checkpoint, last_tmp)
        last_tmp.replace(last_path)

        # Save best model for inference as `model.pt` (default).
        if val_metrics["emb_mean_sim"] > best_val_sim:
            best_val_sim = float(val_metrics["emb_mean_sim"])
            best_epoch = epoch
            checkpoint["best_epoch"] = best_epoch
            checkpoint["best_val_sim"] = best_val_sim

            model_path = checkpoint_dir / "model.pt"
            model_tmp = checkpoint_dir / "model.pt.tmp"
            torch.save(checkpoint, model_tmp)
            model_tmp.replace(model_path)
            print(f"  Updated model.pt (emb_sim={best_val_sim:.4f})")

        # Always write a small progress file so it's clear what epoch is running.
        state_path = checkpoint_dir / "latest_state.json"
        tmp_path = checkpoint_dir / "latest_state.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_emb_mean_sim": best_val_sim,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                f,
                indent=2,
            )
        tmp_path.replace(state_path)
    
    print(f"\nTraining complete!")
    print(f"Best embedding similarity: {best_val_sim:.4f}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Multi-task TRM-UMLS training")
    
    # Data paths
    parser.add_argument("--embedding-dir", type=Path,
                        default=base_dir / "data" / "embeddings")
    parser.add_argument("--assertion-data", type=Path,
                        default=base_dir / "data" / "assertion_train.json")
    parser.add_argument("--subject-data", type=Path,
                        default=base_dir / "data" / "subject_train.json")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=base_dir / "checkpoints")
    parser.add_argument("--v1-checkpoint", type=Path, default=None,
                        help="Path to V1 checkpoint to initialize from")
    parser.add_argument("--resume", action="store_true", help="Resume from a saved checkpoint")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to resume from (defaults to checkpoint-dir/model_last.pt)",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize model weights from a checkpoint (model only; epoch/optimizer reset).",
    )
    
    # Model architecture
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--h-cycles", type=int, default=3)
    parser.add_argument("--l-cycles", type=int, default=4)
    parser.add_argument("--classifier-hidden-size", type=int, default=128)
    parser.add_argument("--classifier-dropout", type=float, default=0.1)
    
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for embedding losses")
    parser.add_argument(
        "--embedding-target",
        type=str,
        choices=["synonym", "concept"],
        default="concept",
        help="Distillation target: 'synonym' matches teacher(synonym); 'concept' matches teacher(prefterm concept center).",
    )
    parser.add_argument(
        "--hard-negs",
        type=Path,
        default=None,
        help="Optional: concept_neighbors.npy (hard negatives) aligned to umls_embeddings.npy rows",
    )
    parser.add_argument("--hard-neg-k", type=int, default=8, help="How many hard negatives to use per example")
    parser.add_argument(
        "--hard-neg-mix",
        type=float,
        default=0.0,
        help="0 disables; otherwise mixes hard-neg loss into embedding loss: (1-mix)*inbatch + mix*hard",
    )
    
    # Loss weights
    parser.add_argument("--emb-weight", type=float, default=0.6)
    parser.add_argument("--assert-weight", type=float, default=0.25)
    parser.add_argument("--subj-weight", type=float, default=0.15)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    main(args)
