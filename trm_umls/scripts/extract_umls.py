#!/usr/bin/env python3
"""
Extract UMLS data from KidneyStone_SDOH.script (HSQLDB dump)

Extracts:
- CUI_TERMS: (CUI, RINDEX, TCOUNT, TEXT, RWORD) - synonym mappings
- PREFTERM: (CUI, PREFTERM) - preferred terms
- TUI: (CUI, TUI) - semantic type mappings

Output: JSON files for downstream embedding + training
"""

import re
import json
from pathlib import Path
from collections import defaultdict
import argparse


def parse_insert_statement(line: str, table_name: str) -> list[str] | None:
    """Parse a single INSERT statement and extract raw values."""
    pattern = rf"INSERT INTO {table_name} VALUES\((.*)\)"
    match = re.match(pattern, line, re.IGNORECASE)
    if not match:
        return None
    
    values_str = match.group(1)
    # Parse values - handle quoted strings with commas
    values = []
    current = ""
    in_quotes = False
    
    for char in values_str:
        if char == "'" and not in_quotes:
            in_quotes = True
        elif char == "'" and in_quotes:
            in_quotes = False
        elif char == "," and not in_quotes:
            values.append(current.strip().strip("'"))
            current = ""
            continue
        current += char
    
    if current:
        values.append(current.strip().strip("'"))
    
    return values


def extract_umls_data(script_path: Path, output_dir: Path):
    """Extract all UMLS tables from the script file."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data containers
    cui_terms = []  # (CUI, RINDEX, TCOUNT, TEXT, RWORD)
    prefterms = {}  # CUI -> preferred term
    tui_mappings = defaultdict(list)  # CUI -> list of TUIs
    
    print(f"Reading {script_path}...")
    print("This may take a few minutes for 5.5M lines...")
    
    line_count = 0
    cui_terms_count = 0
    prefterm_count = 0
    tui_count = 0
    
    with open(script_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line_count += 1
            
            if line_count % 500000 == 0:
                print(f"  Processed {line_count:,} lines...")
            
            line = line.strip()
            
            # Parse CUI_TERMS
            if line.startswith("INSERT INTO CUI_TERMS"):
                values = parse_insert_statement(line, "CUI_TERMS")
                if values and len(values) >= 5:
                    try:
                        cui = int(values[0])
                        rindex = int(values[1])
                        tcount = int(values[2])
                        text = values[3]
                        rword = values[4]
                        cui_terms.append({
                            "cui": cui,
                            "rindex": rindex,
                            "tcount": tcount,
                            "text": text,
                            "rword": rword
                        })
                        cui_terms_count += 1
                    except (ValueError, IndexError):
                        pass
            
            # Parse PREFTERM
            elif line.startswith("INSERT INTO PREFTERM"):
                values = parse_insert_statement(line, "PREFTERM")
                if values and len(values) >= 2:
                    try:
                        cui = int(values[0])
                        prefterm = values[1]
                        prefterms[cui] = prefterm
                        prefterm_count += 1
                    except (ValueError, IndexError):
                        pass
            
            # Parse TUI
            elif line.startswith("INSERT INTO TUI"):
                values = parse_insert_statement(line, "TUI")
                if values and len(values) >= 2:
                    try:
                        cui = int(values[0])
                        tui = int(values[1])
                        tui_mappings[cui].append(tui)
                        tui_count += 1
                    except (ValueError, IndexError):
                        pass
    
    print(f"\nExtraction complete:")
    print(f"  CUI_TERMS: {cui_terms_count:,} entries")
    print(f"  PREFTERM: {prefterm_count:,} entries")
    print(f"  TUI: {tui_count:,} entries")
    
    # Save as JSON (can convert to Parquet later if needed)
    print("\nSaving extracted data...")
    
    # Save CUI_TERMS in batches (too large for single JSON)
    batch_size = 500000
    for i in range(0, len(cui_terms), batch_size):
        batch = cui_terms[i:i+batch_size]
        batch_file = output_dir / f"cui_terms_batch_{i//batch_size}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch, f)
        print(f"  Saved {batch_file.name}")
    
    # Save PREFTERM
    prefterm_file = output_dir / "prefterms.json"
    with open(prefterm_file, 'w') as f:
        json.dump(prefterms, f)
    print(f"  Saved prefterms.json ({len(prefterms):,} terms)")
    
    # Save TUI mappings
    tui_file = output_dir / "tui_mappings.json"
    with open(tui_file, 'w') as f:
        json.dump({str(k): v for k, v in tui_mappings.items()}, f)
    print(f"  Saved tui_mappings.json ({len(tui_mappings):,} CUIs)")
    
    # Create a consolidated concepts file for embedding generation
    print("\nCreating consolidated concepts file for embedding...")
    concepts = []
    for cui, prefterm in prefterms.items():
        concepts.append({
            "cui": cui,
            "prefterm": prefterm,
            "tuis": tui_mappings.get(cui, [])
        })
    
    concepts_file = output_dir / "concepts.json"
    with open(concepts_file, 'w') as f:
        json.dump(concepts, f)
    print(f"  Saved concepts.json ({len(concepts):,} unique concepts)")
    
    # Create synonym lookup for training
    print("\nCreating synonym lookup for training...")
    cui_to_synonyms = defaultdict(list)
    for entry in cui_terms:
        cui_to_synonyms[entry["cui"]].append(entry["text"])
    
    synonyms_file = output_dir / "cui_synonyms.json"
    with open(synonyms_file, 'w') as f:
        json.dump({str(k): v for k, v in cui_to_synonyms.items()}, f)
    print(f"  Saved cui_synonyms.json ({len(cui_to_synonyms):,} CUIs with synonyms)")
    
    # Summary stats
    total_synonyms = sum(len(v) for v in cui_to_synonyms.values())
    avg_synonyms = total_synonyms / len(cui_to_synonyms) if cui_to_synonyms else 0
    
    print(f"\n=== Summary ===")
    print(f"Unique CUIs: {len(prefterms):,}")
    print(f"Total synonyms: {total_synonyms:,}")
    print(f"Average synonyms per CUI: {avg_synonyms:.1f}")
    print(f"Output directory: {output_dir}")
    
    return {
        "cui_count": len(prefterms),
        "synonym_count": total_synonyms,
        "avg_synonyms": avg_synonyms
    }


if __name__ == "__main__":
    trm_umls_dir = Path(__file__).resolve().parents[1]
    repo_root = trm_umls_dir.parent

    parser = argparse.ArgumentParser(description="Extract UMLS data from HSQLDB script")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=repo_root / "KidneyStone_SDOH.script",
        help="Path to the HSQLDB script file"
    )
    parser.add_argument(
        "--output", "-o", 
        type=Path,
        default=trm_umls_dir / "data" / "umls",
        help="Output directory for extracted data"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        exit(1)
    
    extract_umls_data(args.input, args.output)
