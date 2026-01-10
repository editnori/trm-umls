#!/usr/bin/env python3
"""
Generate subject classification training data.

This identifies whether a medical mention refers to:
- PATIENT: The patient themselves
- FAMILY: Family member (family history)
- OTHER: Someone else (e.g., donor, other person mentioned)
"""

import json
import re
from pathlib import Path
from typing import List, Dict
import random
import argparse


SUBJECT_LABELS = ["PATIENT", "FAMILY", "OTHER"]


def generate_synthetic_examples() -> List[Dict]:
    """Generate synthetic subject examples with clear patterns."""
    
    conditions = [
        "diabetes", "hypertension", "heart disease", "cancer",
        "stroke", "kidney disease", "asthma", "COPD",
        "arthritis", "depression", "anxiety", "dementia",
        "Alzheimer's disease", "Parkinson's disease", "epilepsy",
        "breast cancer", "lung cancer", "colon cancer",
        "coronary artery disease", "heart failure",
    ]
    
    templates = {
        "PATIENT": [
            "Patient has {condition}.",
            "The patient was diagnosed with {condition}.",
            "History of {condition}.",
            "Known {condition}.",
            "Active {condition}.",
            "Patient presents with {condition}.",
            "Currently being treated for {condition}.",
            "Diagnosed with {condition} in 2020.",
            "Past medical history significant for {condition}.",
            "PMH: {condition}.",
            "Medical history includes {condition}.",
            "The patient has a history of {condition}.",
            "He has {condition}.",
            "She has {condition}.",
            "Patient reports {condition}.",
        ],
        "FAMILY": [
            "Family history of {condition}.",
            "Father has {condition}.",
            "Mother has {condition}.",
            "Sister has {condition}.",
            "Brother has {condition}.",
            "Grandfather had {condition}.",
            "Grandmother had {condition}.",
            "Family history significant for {condition}.",
            "FH: {condition}.",
            "Maternal history of {condition}.",
            "Paternal history of {condition}.",
            "Parent with {condition}.",
            "Sibling with {condition}.",
            "Mother died of {condition}.",
            "Father died of {condition}.",
            "Strong family history of {condition}.",
            "Multiple family members with {condition}.",
        ],
        "OTHER": [
            "Donor history of {condition}.",
            "Caregiver has {condition}.",
            "Roommate diagnosed with {condition}.",
            "Contact with person who has {condition}.",
            "Exposed to individual with {condition}.",
            "Partner has {condition}.",
            "Spouse has {condition}.",
            "Friend died of {condition}.",
            "Colleague has {condition}.",
            "Healthcare worker with {condition}.",
        ],
    }
    
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


def extract_from_clinical_notes(notes_dir: Path, max_notes: int = 200) -> List[Dict]:
    """Extract subject examples from clinical notes based on section headers."""
    examples = []
    
    # Section patterns that indicate subject
    family_patterns = [
        r'family\s*history',
        r'FH:',
        r'FHx:',
        r'mother\s+(?:has|had|with)',
        r'father\s+(?:has|had|with)',
        r'sister\s+(?:has|had|with)',
        r'brother\s+(?:has|had|with)',
        r'parent\s+(?:has|had|with)',
    ]
    
    patient_patterns = [
        r'past\s*medical\s*history',
        r'PMH:',
        r'PMHx:',
        r'history\s*of\s*present\s*illness',
        r'HPI:',
        r'chief\s*complaint',
        r'CC:',
        r'assessment\s*(?:and|/)\s*plan',
        r'A/P:',
        r'patient\s+(?:has|had|presents|reports)',
    ]
    
    note_files = list(notes_dir.glob("**/*.txt"))[:max_notes]
    
    for note_file in note_files:
        try:
            with open(note_file, 'r', errors='ignore') as f:
                text = f.read()
            
            # Split into lines/sentences
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if len(line) < 10 or len(line) > 200:
                    continue
                
                # Check for family patterns
                is_family = any(re.search(p, line, re.IGNORECASE) for p in family_patterns)
                is_patient = any(re.search(p, line, re.IGNORECASE) for p in patient_patterns)
                
                if is_family and not is_patient:
                    examples.append({
                        "text": line,
                        "label": "FAMILY",
                        "source": "clinical_note"
                    })
                elif is_patient and not is_family:
                    examples.append({
                        "text": line,
                        "label": "PATIENT",
                        "source": "clinical_note"
                    })
                    
        except Exception:
            continue
    
    return examples


def main(args: argparse.Namespace):
    print("Generating subject classification training data...")
    
    # Generate synthetic examples
    print("\nGenerating synthetic examples...")
    synthetic = generate_synthetic_examples()
    print(f"  Generated {len(synthetic)} synthetic examples")
    
    # Extract from clinical notes
    if args.notes_dir and args.notes_dir.exists():
        print("\nExtracting from clinical notes...")
        clinical = extract_from_clinical_notes(args.notes_dir, max_notes=args.max_notes)
        print(f"  Extracted {len(clinical)} examples from clinical notes")
    else:
        clinical = []
    
    # Combine
    all_examples = synthetic + clinical
    random.shuffle(all_examples)
    
    # Count distribution
    from collections import Counter
    label_counts = Counter(ex["label"] for ex in all_examples)
    
    print(f"\nTotal examples: {len(all_examples)}")
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")
    
    # Save
    output_path = args.output
    with open(output_path, 'w') as f:
        json.dump(all_examples, f)
    
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    trm_umls_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate subject classification training data")
    parser.add_argument("--notes-dir", type=Path, default=None,
                        help="Optional directory with clinical notes (.txt) to mine for extra examples")
    parser.add_argument("--max-notes", type=int, default=300)
    parser.add_argument("--output", type=Path, default=trm_umls_dir / "data" / "subject_train.json")
    main(parser.parse_args())
