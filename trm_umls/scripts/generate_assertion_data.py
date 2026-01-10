#!/usr/bin/env python3
"""
Generate assertion training data using clinical-assertion-negation-bert as teacher.

This script:
1. Loads clinical notes from SD5000
2. Extracts medical entities and their context
3. Uses clinical-assertion-negation-bert to label them as PRESENT/ABSENT/POSSIBLE
4. Saves the training data for TRM multi-task learning
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import random
import argparse


# Assertion labels
ASSERTION_LABELS = ["PRESENT", "ABSENT", "POSSIBLE"]


def load_assertion_model(device: str = "cuda"):
    """Load the clinical assertion BERT model."""
    model_name = "bvanaken/clinical-assertion-negation-bert"
    print(f"Loading assertion model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def extract_medical_phrases(text: str) -> List[Dict]:
    """
    Extract candidate medical phrases from clinical text.
    Uses simple pattern matching for common medical term patterns.
    """
    # Common patterns for medical terms
    patterns = [
        # Conditions/diagnoses
        r'\b(hypertension|diabetes|pneumonia|sepsis|cancer|infection|disease|syndrome|disorder|failure|insufficiency)\b',
        # Symptoms
        r'\b(pain|fever|cough|nausea|vomiting|bleeding|swelling|weakness|fatigue|dyspnea|edema)\b',
        # Findings
        r'\b(effusion|consolidation|mass|lesion|nodule|calcification|stenosis|obstruction)\b',
        # Body parts with conditions
        r'\b(cardiac|pulmonary|renal|hepatic|cerebral|abdominal)\s+\w+',
        # Drug names (simplified)
        r'\b(aspirin|metoprolol|lisinopril|metformin|insulin|warfarin|heparin)\b',
    ]
    
    phrases = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Get context window
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            
            phrases.append({
                "phrase": match.group(0),
                "context": context,
                "start": match.start(),
                "end": match.end()
            })
    
    return phrases


def get_assertion_label(model, tokenizer, context: str, entity: str, device: str) -> Tuple[str, float]:
    """
    Get assertion label for an entity in context using the teacher model.
    
    The model expects input with [entity] markers around the target term.
    """
    # Format input with entity markers
    marked_context = context.replace(entity, f"[entity] {entity} [entity]", 1)
    
    # Tokenize
    inputs = tokenizer(
        marked_context,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_idx = probs.argmax().item()
        confidence = probs[0, pred_idx].item()
    
    label = ASSERTION_LABELS[pred_idx]
    return label, confidence


def generate_synthetic_examples() -> List[Dict]:
    """
    Generate synthetic assertion examples with clear patterns.
    These provide cleaner training signal than noisy clinical notes.
    """
    templates = {
        "PRESENT": [
            "Patient has {condition}.",
            "Diagnosed with {condition}.",
            "History of {condition}.",
            "Known {condition}.",
            "Active {condition}.",
            "Patient presents with {condition}.",
            "Positive for {condition}.",
            "Confirmed {condition}.",
            "{condition} is present.",
            "Evidence of {condition}.",
        ],
        "ABSENT": [
            "Patient denies {condition}.",
            "No {condition}.",
            "Denies {condition}.",
            "Negative for {condition}.",
            "No evidence of {condition}.",
            "Without {condition}.",
            "Ruled out {condition}.",
            "{condition} absent.",
            "No signs of {condition}.",
            "Patient does not have {condition}.",
        ],
        "POSSIBLE": [
            "Possible {condition}.",
            "Suspected {condition}.",
            "Rule out {condition}.",
            "Cannot exclude {condition}.",
            "Consider {condition}.",
            "Likely {condition}.",
            "Probable {condition}.",
            "Questionable {condition}.",
            "May have {condition}.",
            "{condition} is uncertain.",
        ],
    }
    
    conditions = [
        "hypertension", "diabetes", "pneumonia", "heart failure",
        "chronic kidney disease", "COPD", "atrial fibrillation",
        "coronary artery disease", "stroke", "pulmonary embolism",
        "deep vein thrombosis", "anemia", "hypothyroidism",
        "hyperlipidemia", "obesity", "depression", "anxiety",
        "asthma", "arthritis", "osteoporosis", "dementia",
        "urinary tract infection", "sepsis", "acute kidney injury",
        "myocardial infarction", "congestive heart failure",
    ]
    
    examples = []
    for label, template_list in templates.items():
        for template in template_list:
            for condition in conditions:
                text = template.format(condition=condition)
                examples.append({
                    "text": text,
                    "entity": condition,
                    "label": label,
                    "source": "synthetic"
                })
    
    return examples


def process_clinical_notes(
    notes_dir: Path,
    model,
    tokenizer,
    device: str,
    max_notes: int = 100,
    max_examples_per_note: int = 10
) -> List[Dict]:
    """Process clinical notes to extract assertion examples."""
    examples = []
    
    # Get all note files
    note_files = list(notes_dir.glob("**/*.txt"))[:max_notes]
    
    print(f"Processing {len(note_files)} clinical notes...")
    
    for note_file in tqdm(note_files):
        try:
            with open(note_file, 'r', errors='ignore') as f:
                text = f.read()
            
            # Extract medical phrases
            phrases = extract_medical_phrases(text)[:max_examples_per_note]
            
            for phrase_info in phrases:
                try:
                    label, confidence = get_assertion_label(
                        model, tokenizer,
                        phrase_info["context"],
                        phrase_info["phrase"],
                        device
                    )
                    
                    if confidence > 0.7:  # Only keep high-confidence examples
                        examples.append({
                            "text": phrase_info["context"],
                            "entity": phrase_info["phrase"],
                            "label": label,
                            "confidence": confidence,
                            "source": "clinical_note"
                        })
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
    
    return examples


def main(args: argparse.Namespace):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load teacher model
    model, tokenizer = load_assertion_model(device)
    
    # Generate synthetic examples (clean signal)
    print("\nGenerating synthetic examples...")
    synthetic_examples = generate_synthetic_examples()
    print(f"  Generated {len(synthetic_examples)} synthetic examples")
    
    # Process clinical notes (real-world signal)
    if args.notes_dir and args.notes_dir.exists():
        print("\nProcessing clinical notes...")
        clinical_examples = process_clinical_notes(
            args.notes_dir, model, tokenizer, device,
            max_notes=args.max_notes,
            max_examples_per_note=args.max_examples_per_note
        )
        print(f"  Extracted {len(clinical_examples)} examples from clinical notes")
    else:
        clinical_examples = []
    
    # Combine and balance
    all_examples = synthetic_examples + clinical_examples
    random.shuffle(all_examples)
    
    # Count labels
    label_counts = {}
    for ex in all_examples:
        label_counts[ex["label"]] = label_counts.get(ex["label"], 0) + 1
    
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Save
    output_path = args.output
    with open(output_path, 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    print(f"\nSaved {len(all_examples)} examples to {output_path}")


if __name__ == "__main__":
    trm_umls_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate assertion training data")
    parser.add_argument("--notes-dir", type=Path, default=None,
                        help="Optional directory with clinical notes (.txt) to mine for extra examples")
    parser.add_argument("--max-notes", type=int, default=200)
    parser.add_argument("--max-examples-per-note", type=int, default=5)
    parser.add_argument("--device", type=str, default=None,
                        help="Device for teacher model (e.g., 'cuda', 'cpu'); defaults to auto-detect")
    parser.add_argument("--output", type=Path, default=trm_umls_dir / "data" / "assertion_train.json")
    main(parser.parse_args())
