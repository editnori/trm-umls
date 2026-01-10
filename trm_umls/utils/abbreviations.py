#!/usr/bin/env python3
"""
Medical abbreviation expansion utility.

Expands common medical abbreviations (HTN, COPD, CHF, etc.) to their
full forms before embedding, improving concept matching accuracy.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class AbbreviationExpander:
    """Expands medical abbreviations to their full forms."""
    
    # Minimum length for abbreviations to avoid false positives
    MIN_ABBREV_LENGTH = 2
    
    def __init__(self, abbrev_path: Optional[Path] = None, min_length: int = 2):
        """
        Initialize the expander with abbreviation mappings.
        
        Args:
            abbrev_path: Path to abbreviations.json. If None, uses default.
            min_length: Minimum abbreviation length to expand (default 2)
        """
        if abbrev_path is None:
            abbrev_path = Path(__file__).parent.parent / "data" / "abbreviations.json"
        
        with open(abbrev_path, 'r') as f:
            self.abbreviations: Dict[str, str] = json.load(f)
        
        # Create case-insensitive lookup, filtering by min length
        self.abbrev_upper = {
            k.upper(): v 
            for k, v in self.abbreviations.items() 
            if len(k) >= min_length
        }
        
        # Safe 2-letter abbreviations (common medical terms)
        self.safe_2letter = {
            'MI', 'PE', 'GI', 'IV', 'IM', 'PO', 'SC', 'BP', 'HR', 'RR',
            'OA', 'RA', 'ER', 'ED', 'CT', 'MR', 'US', 'EF', 'DM'
        }
        
        # Build regex pattern for word-boundary matching
        # Sort by length (longest first) to match longer abbreviations first
        # Include abbreviations >= 3 chars OR in safe_2letter list
        sorted_abbrevs = sorted(
            [a for a in self.abbrev_upper.keys() 
             if len(a) >= 3 or a in self.safe_2letter], 
            key=len, 
            reverse=True
        )
        escaped = [re.escape(a) for a in sorted_abbrevs]
        self.pattern = re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)
    
    def expand(self, text: str, return_mappings: bool = False) -> str | Tuple[str, List[Dict]]:
        """
        Expand abbreviations in text.
        
        Args:
            text: Input text with potential abbreviations
            return_mappings: If True, also return list of expansions made
            
        Returns:
            Expanded text, or (expanded_text, mappings) if return_mappings=True
        """
        mappings = []
        
        def replace_fn(match):
            abbrev = match.group(0)
            abbrev_upper = abbrev.upper()
            if abbrev_upper in self.abbrev_upper:
                # Avoid false positives for 2-letter abbreviations by requiring ALL CAPS.
                if len(abbrev_upper) == 2 and abbrev_upper in self.safe_2letter and abbrev != abbrev_upper:
                    return abbrev
                expansion = self.abbrev_upper[abbrev_upper]
                mappings.append({
                    "original": abbrev,
                    "expanded": expansion,
                    "start": match.start(),
                    "end": match.end()
                })
                return expansion
            return abbrev
        
        expanded = self.pattern.sub(replace_fn, text)
        
        if return_mappings:
            return expanded, mappings
        return expanded
    
    def expand_term(self, term: str) -> str:
        """
        Expand a single term if it's an abbreviation.
        
        Args:
            term: Single term to expand
            
        Returns:
            Expanded term or original if not an abbreviation
        """
        term_upper = term.strip().upper()
        return self.abbrev_upper.get(term_upper, term)
    
    def is_abbreviation(self, term: str) -> bool:
        """Check if a term is a known abbreviation."""
        return term.strip().upper() in self.abbrev_upper
    
    def get_expansion(self, abbrev: str) -> Optional[str]:
        """Get the expansion for an abbreviation, or None if not found."""
        return self.abbrev_upper.get(abbrev.strip().upper())
    
    def __len__(self) -> int:
        return len(self.abbreviations)
    
    def __contains__(self, item: str) -> bool:
        return self.is_abbreviation(item)


def test_expander():
    """Test the abbreviation expander."""
    expander = AbbreviationExpander()
    print(f"Loaded {len(expander)} abbreviations")
    
    # Test cases
    test_cases = [
        "Patient has HTN and DM2.",
        "History of COPD, CHF, and CAD.",
        "Denies CVA or UTI.",
        "Labs show elevated BUN and Cr.",
        "Family history of MI and PE.",
        "Diagnosed with GERD and CKD stage 3.",
    ]
    
    print("\nExpansion tests:")
    print("-" * 60)
    
    for text in test_cases:
        expanded, mappings = expander.expand(text, return_mappings=True)
        print(f"Original: {text}")
        print(f"Expanded: {expanded}")
        if mappings:
            print(f"Mappings: {mappings}")
        print()


if __name__ == "__main__":
    test_expander()
