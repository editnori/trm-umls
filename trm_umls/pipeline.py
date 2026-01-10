#!/usr/bin/env python3
"""
TRM-UMLS Pipeline: Unified concept extraction with cTAKES-like output.

Provides a simple API for extracting medical concepts from clinical text:
- CUI (Concept Unique Identifier)
- Preferred Term
- TUI (Semantic Type) - from CUI lookup
- Semantic Group
- Assertion (PRESENT/ABSENT/POSSIBLE)
- Subject (PATIENT/FAMILY/OTHER)
"""

import json
import re
import argparse
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import numpy as np
import torch
import faiss

try:
    from .models.trm_text_encoder import TRMTextEncoder, TRMTextEncoderConfig
    from .utils.abbreviations import AbbreviationExpander
except ImportError:  # pragma: no cover
    from models.trm_text_encoder import TRMTextEncoder, TRMTextEncoderConfig
    from utils.abbreviations import AbbreviationExpander


# Label mappings
ASSERTION_LABELS = ["PRESENT", "ABSENT", "POSSIBLE"]
SUBJECT_LABELS = ["PATIENT", "FAMILY", "OTHER"]

# Semantic group mappings
SEMANTIC_GROUPS = {
    "ACTI": "Activities & Behaviors",
    "ANAT": "Anatomy",
    "CHEM": "Chemicals & Drugs",
    "CONC": "Concepts & Ideas",
    "DEVI": "Devices",
    "DISO": "Disorders",
    "GENE": "Genes & Molecular Sequences",
    "GEOG": "Geographic Areas",
    "LIVB": "Living Beings",
    "OBJC": "Objects",
    "OCCU": "Occupations",
    "ORGA": "Organizations",
    "PHEN": "Phenomena",
    "PHYS": "Physiology",
    "PROC": "Procedures",
}

# TUI to semantic group mapping (abridged - key clinical TUIs)
TUI_TO_GROUP = {
    47: "DISO",   # Disease or Syndrome
    184: "DISO",  # Sign or Symptom
    33: "DISO",   # Finding
    191: "DISO",  # Neoplastic Process
    121: "CHEM",  # Pharmacologic Substance
    200: "CHEM",  # Clinical Drug
    109: "CHEM",  # Organic Chemical
    116: "CHEM",  # Amino Acid, Peptide, or Protein
    23: "ANAT",   # Body Part, Organ, or Organ Component
    29: "ANAT",   # Body Location or Region
    60: "PROC",   # Diagnostic Procedure
    61: "PROC",   # Therapeutic or Preventive Procedure
    59: "PROC",   # Laboratory Procedure
    34: "PHEN",   # Laboratory or Test Result
}


@dataclass
class ConceptExtraction:
    """A single extracted concept."""
    text: str                    # Original text span
    expanded_text: str           # After abbreviation expansion
    normalized_text: str         # Text used for embedding/lookup (may strip modifiers)
    cui: str                     # UMLS CUI (e.g., "C0020538")
    preferred_term: str          # Preferred term from UMLS
    tui: str                     # Semantic type (e.g., "T047")
    semantic_type: str           # Semantic type name
    semantic_group: str          # Semantic group (e.g., "DISO")
    semantic_group_name: str     # Semantic group name
    assertion: str               # PRESENT/ABSENT/POSSIBLE
    subject: str                 # PATIENT/FAMILY/OTHER
    score: float                 # Confidence score
    rerank_score: float = 0.0    # Score after reranking (if enabled)
    severity: Optional[str] = None
    laterality: Optional[str] = None
    temporality: Optional[str] = None
    start: int = -1              # Start position in original text
    end: int = -1                # End position in original text
    candidates: Optional[List[Dict[str, Any]]] = None  # Optional top-k candidates for debugging / review


class TRMUMLSPipeline:
    """
    Unified pipeline for medical concept extraction.
    
    Example:
        pipeline = TRMUMLSPipeline.load("checkpoints/model.pt")
        results = pipeline.extract("Patient denies HTN. Family history of DM.")
    """
    
    def __init__(
        self,
        model: TRMTextEncoder,
        tokenizer,
        index: faiss.Index,
        cuis: np.ndarray,
        prefterms: List[str],
        tui_mappings: Dict[str, List[int]],
        abbrev_expander: AbbreviationExpander,
        device: torch.device,
        mrrel_indptr: Optional[np.ndarray] = None,
        mrrel_indices: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.index = index
        self.cuis = cuis
        self.prefterms = prefterms
        self.tui_mappings = tui_mappings
        self.abbrev_expander = abbrev_expander
        self.device = device
        self.mrrel_indptr = mrrel_indptr
        self.mrrel_indices = mrrel_indices
        
        self.model.eval()

    _tok_re = re.compile(r"[A-Za-z0-9]+")
    _has_alpha_re = re.compile(r"[A-Za-z]")
    _severity_re = re.compile(
        r"\b(mild(?:ly)?|moderate(?:ly)?|severe(?:ly)?|marked(?:ly)?|profound(?:ly)?|minimal(?:ly)?|slight(?:ly)?|significant(?:ly)?|trace|trivial)\b",
        flags=re.IGNORECASE,
    )
    _laterality_re = re.compile(
        r"\b(left|right|bilateral|unilateral)\b|\b(b\/l|b\\l)\b", flags=re.IGNORECASE
    )
    _temporality_re = re.compile(r"\b(acute|chronic|subacute)\b", flags=re.IGNORECASE)
    _lead_stop_re = re.compile(
        r"^(?:of|for|with|without|in|on|at|to|from)\s+(?:the|a|an)\s+",
        flags=re.IGNORECASE,
    )
    _tail_determiner_re = re.compile(r"\s+(?:the|a|an)\s*$", flags=re.IGNORECASE)
    _tail_incomplete_re = re.compile(
        r"\s+(?:of|for|with|without|in|on|at|to|from)\s*$",
        flags=re.IGNORECASE,
    )
    _tail_incomplete_with_det_re = re.compile(
        r"\s+(?:of|for|with|without|in|on|at|to|from)\s+(?:the|a|an)\s*$",
        flags=re.IGNORECASE,
    )
    _tail_field_label_re = re.compile(
        r"\s+(?:start\s+date|end\s+date|stop\s+date|types?)\s*$",
        flags=re.IGNORECASE,
    )

    _stop_mentions = {
        "other",
        "others",
        "all other",
        "all others",
        "more",
        "negative",
        "positive",
        "normal",
        "abnormal",
        "lab results",
        "laboratory results",
        "component value date",
        "component value",
        "radiology impressions",
        "radiology impression",
        "impression",
        "impressions",
        "daily labs",
        "none",
        "unknown",
        "n/a",
        "na",
        "yes",
        "no",
        "not applicable",
    }
    _other_prefix_re = re.compile(r"^\s*other\s*:\s*", flags=re.IGNORECASE)
    _section_headers = {
        "assessment",
        "assessment/plan",
        "assessment and plan",
        "a/p",
        "plan",
        "diagnosis",
        "diagnoses",
        "history",
        "medical history",
        "past medical history",
        "past surgical history",
        "pmh",
        "hpi",
        "ros",
        "review of systems",
        "medications",
        "meds",
        "allergies",
        "family history",
        "social history",
        "surgical history",
        "surgical hx",
        "physical exam",
        "exam",
        "labs",
        "imaging",
        "problem list",
        "chief complaint",
        "cc",
        "disposition",
        "consults",
        "results",
        "skin",
        "general",
        "breath sounds",
        "musculoskeletal",
        "tenderness",
        "respiratory",
        "cardiovascular",
        "genitourinary",
        "gastrointestinal",
        "neurologic",
        "neurological",
        "neuro",
        "psych",
        "psychiatric",
        "vitals",
        "planned procedures",
        "procedures",
        "procedure laterality date",
        "procedure laterality",
        "data review",
        "comment",
        "impression",
        "impressions",
        "lab results",
        "daily labs",
        "radiology impressions",
        "ekg",
        "day",
        "type",
        "types",
        "start date",
        "end date",
        "stop date",
        "liver disease",
    }
    _short_span_allowlist = {
        "EGD",
        "CABG",
        "HGB",
        "LHC",
        "HTN",
        "DM",
        "CHF",
        "COPD",
    }
    _short_span_min_chars = 4
    _header_prefix_re = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /\\\\-]{0,40})\s*:\s*")
    _mid_header_re = re.compile(
        r"\s+([A-Za-z][A-Za-z0-9/\\\\-]*(?: [A-Za-z0-9/\\\\-]+){0,6})\s*:\s*"
    )
    _lab_num_re = re.compile(r"\b(?:lab|laboratory)\s*\d+\b", flags=re.IGNORECASE)

    _prefterm_penalty_rules = [
        (re.compile(r"\bctcae\b", flags=re.IGNORECASE), -0.05),
        (re.compile(r"\brndx\b", flags=re.IGNORECASE), -0.04),
        (re.compile(r"\breported\s+by\s+patient\b", flags=re.IGNORECASE), -0.03),
        (re.compile(r"\bhow\s+often\b", flags=re.IGNORECASE), -0.03),
        (re.compile(r"\bquestion\b", flags=re.IGNORECASE), -0.03),
        (re.compile(r"\bwere\s+you\b", flags=re.IGNORECASE), -0.02),
        (re.compile(r"\bpatient\b", flags=re.IGNORECASE), -0.02),
        (re.compile(r"^when\b", flags=re.IGNORECASE), -0.01),
    ]

    _default_group_bias = {
        "DISO": 0.010,
        "PROC": 0.006,
        "CHEM": 0.004,
        "PHEN": 0.003,
        "ANAT": 0.002,
        "CONC": -0.010,
        "UNKN": -0.010,
    }
    _default_tui_bias = {
        "T184": 0.010,  # Sign or Symptom
        "T047": 0.008,  # Disease or Syndrome
        "T061": 0.006,  # Therapeutic or Preventive Procedure
        "T060": 0.006,  # Diagnostic Procedure
        "T033": -0.004,  # Finding (often includes templated/UI concepts)
        "T170": -0.010,  # Intellectual Product (questions, UI prompts)
    }

    def _token_set(self, text: str) -> Set[str]:
        return {m.group(0).lower() for m in self._tok_re.finditer(text or "")}

    def _jaccard(self, a: str, b: str) -> float:
        sa = self._token_set(a)
        sb = self._token_set(b)
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _normalize_header(self, text: str) -> str:
        t = (text or "").strip().lower()
        if not t:
            return ""
        t = re.sub(r"\s*/\s*", "/", t)
        t = t.rstrip(":")
        t = re.sub(r"\s+", " ", t)
        return t

    def _is_section_header(self, text: str) -> bool:
        norm = self._normalize_header(text)
        if not norm:
            return False
        return norm in self._section_headers

    def _allow_short_span(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw)
        if not cleaned:
            return False
        if len(cleaned) >= int(self._short_span_min_chars):
            return True
        return cleaned.upper() in self._short_span_allowlist

    def _extract_modifiers(self, text: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        """
        Extract simple modifiers and return (normalized_text, severity, laterality, temporality).

        This is intentionally lightweight and rule-based. It helps when CUIs don't encode
        severity/laterality explicitly and reduces embedding noise by stripping modifiers.
        """
        if not text:
            return "", None, None, None

        severity: Optional[str] = None
        laterality: Optional[str] = None
        temporality: Optional[str] = None

        m = self._severity_re.search(text)
        if m:
            sev = m.group(1).lower()
            if sev.endswith("ly") and sev[:-2] in {"mild", "moderate", "severe", "marked", "profound", "minimal", "slight", "significant"}:
                sev = sev[:-2]
            severity = sev
        m = self._laterality_re.search(text)
        if m:
            laterality = (m.group(1) or m.group(2) or "").lower().replace("\\", "/")
        m = self._temporality_re.search(text)
        if m:
            temporality = m.group(1).lower()

        # Strip modifier tokens for embedding/lookup.
        norm = text
        norm = self._severity_re.sub(" ", norm)
        norm = self._laterality_re.sub(" ", norm)
        norm = self._temporality_re.sub(" ", norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        norm = re.sub(r"^(?:any|some|the|a|an)\s+", "", norm, flags=re.IGNORECASE)
        # Table-ish prefixes/suffixes (dates/values) often hurt linking.
        norm = re.sub(r"^(?:\d{1,4}\s+)+", "", norm).strip()
        norm = re.sub(r"\s+\d+(?:\.\d+)?\s*$", "", norm).strip()
        return norm, severity, laterality, temporality

    def _relation_hits(self, concept_row: int, context_rows: Set[int], max_degree: int) -> int:
        """
        Count MRREL neighbor overlaps between `concept_row` and `context_rows`.

        This is a small, optional disambiguation signal. We cap by `max_degree`
        to avoid broad hub nodes dominating reranking.
        """
        if self.mrrel_indptr is None or self.mrrel_indices is None:
            return 0
        if not context_rows:
            return 0
        r = int(concept_row)
        if r < 0 or r + 1 >= int(self.mrrel_indptr.shape[0]):
            return 0
        start = int(self.mrrel_indptr[r])
        end = int(self.mrrel_indptr[r + 1])
        deg = end - start
        if deg <= 0 or deg > int(max_degree):
            return 0
        neigh = self.mrrel_indices[start:end]
        hits = 0
        for n in neigh:
            if int(n) in context_rows:
                hits += 1
        return hits

    def _prefterm_penalty(self, preferred_term: str, span_text: Optional[str] = None) -> float:
        if not preferred_term:
            return 0.0
        penalty = 0.0
        for pat, w in self._prefterm_penalty_rules:
            if pat.search(preferred_term):
                penalty += float(w)
        if span_text:
            penalty += float(self._short_span_prefterm_penalty(span_text, preferred_term))
        return float(penalty)

    def _short_span_prefterm_penalty(self, span_text: str, preferred_term: str) -> float:
        span = (span_text or "").strip()
        pref = (preferred_term or "").strip()
        if not span or not pref:
            return 0.0
        span_tokens = len(self._token_set(span))
        pref_tokens = len(self._token_set(pref))
        span_len = len(span)
        pref_len = len(pref)

        if span_tokens <= 1 and pref_tokens >= 5:
            return -0.05
        if span_tokens <= 2 and pref_tokens >= 7:
            return -0.04
        if span_len <= 6 and pref_len >= 30:
            return -0.03
        if span_len <= 10 and pref_len >= 45:
            return -0.02
        return 0.0

    @staticmethod
    def _normalize_tui_key(key: str) -> str:
        k = (key or "").strip().upper()
        if not k:
            return ""
        if k.startswith("T") and len(k) == 4 and k[1:].isdigit():
            return k
        if k.isdigit():
            return f"T{int(k):03d}"
        if k.startswith("T") and k[1:].isdigit():
            return f"T{int(k[1:]):03d}"
        return k

    @staticmethod
    def _parse_bias_map(spec: Optional[str], *, kind: str) -> Dict[str, float]:
        """
        Parse comma-separated KEY=FLOAT entries into a dict.

        kind:
          - "group": keys like DISO, PROC, ...
          - "tui": keys like T184 or 184
        """
        if spec is None:
            return {}
        raw = str(spec).strip()
        if not raw:
            return {}

        out: Dict[str, float] = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Invalid {kind} bias entry (expected KEY=FLOAT): {part!r}")
            k, v = part.split("=", 1)
            k = k.strip().upper()
            v = float(v.strip())
            if kind == "tui":
                k = TRMUMLSPipeline._normalize_tui_key(k)
            out[k] = float(v)
        return out
    
    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        index_path: str = None,
        metadata_path: str = None,
        tui_mappings_path: str = None,
        device: str = "cuda",
        mrrel_indptr_path: str = None,
        mrrel_indices_path: str = None,
    ) -> "TRMUMLSPipeline":
        """Load pipeline from checkpoint."""
        base_dir = Path(checkpoint_path).parent.parent
        
        if index_path is None:
            index_path = base_dir / "data" / "embeddings" / "umls_flat.index"
        if metadata_path is None:
            metadata_path = base_dir / "data" / "embeddings" / "embedding_metadata.json"
        if tui_mappings_path is None:
            tui_mappings_path = base_dir / "data" / "umls" / "tui_mappings.json"
        if mrrel_indptr_path is None:
            mrrel_indptr_path = base_dir / "data" / "umls" / "mrrel_indptr.npy"
        if mrrel_indices_path is None:
            mrrel_indices_path = base_dir / "data" / "umls" / "mrrel_indices.npy"
        
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = TRMTextEncoderConfig(**checkpoint["config"])
        model = TRMTextEncoder(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Parameters: {model.count_parameters():,}")
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Load FAISS index
        print(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(str(index_path))
        print(f"  Vectors: {index.ntotal:,}")
        if hasattr(index, "d") and int(config.embedding_dim) != int(index.d):
            raise ValueError(
                "Model/index embedding dim mismatch: "
                f"model embedding_dim={int(config.embedding_dim)}, index.d={int(index.d)}. "
                "Load a checkpoint trained for this index or pass matching --index-path/--metadata-path."
            )
        
        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
        cuis = np.array(metadata["cuis"])
        prefterms = metadata["prefterms"]

        # Basic consistency checks (fail fast with a clear error)
        emb_dim = int(metadata.get("embedding_dim", 0) or 0)
        if emb_dim and hasattr(index, "d") and index.d != emb_dim:
            raise ValueError(f"Index dim ({index.d}) != metadata embedding_dim ({emb_dim})")
        if index.ntotal != len(cuis) or index.ntotal != len(prefterms):
            raise ValueError(
                "Index/metadata mismatch: "
                f"index.ntotal={index.ntotal:,}, len(cuis)={len(cuis):,}, len(prefterms)={len(prefterms):,}"
            )
        
        # Load TUI mappings
        with open(tui_mappings_path) as f:
            tui_mappings = json.load(f)
        
        # Load abbreviation expander
        abbrev_expander = AbbreviationExpander()
        print(f"  Abbreviations: {len(abbrev_expander)}")

        # Load MRREL CSR (optional; only used if relation rerank is enabled)
        mrrel_indptr = None
        mrrel_indices = None
        try:
            indptr_p = Path(mrrel_indptr_path) if mrrel_indptr_path is not None else None
            indices_p = Path(mrrel_indices_path) if mrrel_indices_path is not None else None
            if indptr_p is not None and indices_p is not None and indptr_p.exists() and indices_p.exists():
                mrrel_indptr = np.load(indptr_p, mmap_mode="r")
                mrrel_indices = np.load(indices_p, mmap_mode="r")
                # Basic sanity checks (fail fast if malformed)
                if int(mrrel_indptr.shape[0]) != int(index.ntotal) + 1:
                    raise ValueError(
                        "MRREL CSR size mismatch: "
                        f"len(indptr)={int(mrrel_indptr.shape[0])} vs index.ntotal+1={int(index.ntotal)+1}"
                    )
                if int(mrrel_indptr[-1]) != int(mrrel_indices.shape[0]):
                    raise ValueError(
                        "MRREL CSR mismatch: "
                        f"indptr[-1]={int(mrrel_indptr[-1])} != len(indices)={int(mrrel_indices.shape[0])}"
                    )
                print(
                    f"  MRREL: loaded CSR (edges={int(mrrel_indices.shape[0]):,})",
                )
        except Exception as e:
            print(f"  MRREL: failed to load ({e}); relation rerank will be unavailable")
            mrrel_indptr = None
            mrrel_indices = None
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            index=index,
            cuis=cuis,
            prefterms=prefterms,
            tui_mappings=tui_mappings,
            abbrev_expander=abbrev_expander,
            device=device,
            mrrel_indptr=mrrel_indptr,
            mrrel_indices=mrrel_indices,
        )
    
    def _get_tui_info(self, cui: int) -> Tuple[str, str, str, str]:
        """Get TUI and semantic group info for a CUI."""
        tuis = self.tui_mappings.get(str(cui), [])
        
        if not tuis:
            return "T000", "Unknown", "UNKN", "Unknown"
        
        primary_tui = tuis[0] if isinstance(tuis, list) else tuis
        tui_str = f"T{primary_tui:03d}"
        
        # Get semantic group
        group = TUI_TO_GROUP.get(primary_tui, "CONC")
        group_name = SEMANTIC_GROUPS.get(group, "Concepts & Ideas")
        
        # Get semantic type name (simplified)
        type_names = {
            47: "Disease or Syndrome",
            184: "Sign or Symptom",
            33: "Finding",
            191: "Neoplastic Process",
            121: "Pharmacologic Substance",
            200: "Clinical Drug",
            109: "Organic Chemical",
            23: "Body Part, Organ, or Organ Component",
            60: "Diagnostic Procedure",
            61: "Therapeutic or Preventive Procedure",
            59: "Laboratory Procedure",
            34: "Laboratory or Test Result",
        }
        type_name = type_names.get(primary_tui, "Concept")
        
        return tui_str, type_name, group, group_name

    def _detect_context(self, sentence: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Detect a context prefix that should be stripped before concept matching, plus strong label cues.

        Goal: isolate mention text from clinical framing ("denies", "family history of", etc.).
        """
        rules: List[Tuple[str, Optional[str], Optional[str]]] = [
            (r"^family\s+(?:history|hx)\s+of\s+", "PRESENT", "FAMILY"),
            (r"^family\s+(?:history|hx)\s*[:\\-]?\s*", "PRESENT", "FAMILY"),
            (r"^(?:fhx|fam\s*hx)\s*[:\\-]?\s*", "PRESENT", "FAMILY"),
            (r"^(?:patient|pt|he|she|they)\s+(?:denies|denied)[:\\-]?\s+", "ABSENT", "PATIENT"),
            (r"^(?:denies|denied)[:\\-]?\s+", "ABSENT", "PATIENT"),
            (r"^there\s+(?:is|was|are|were)\s+no\s+", "ABSENT", "PATIENT"),
            (r"^(?:no\s+evidence\s+of|without|negative\s+for)\s+", "ABSENT", "PATIENT"),
            (r"^no\s+", "ABSENT", "PATIENT"),
            (r"^rule\s+out\s+", "POSSIBLE", "PATIENT"),
            (r"^(?:r/o|r\s*/\s*o)\s+", "POSSIBLE", "PATIENT"),
            (r"^(?:possible|suspected|concern\s+for)\s+", "POSSIBLE", "PATIENT"),
            (r"^(?:history|hx)\s+of\s+", None, None),
            (r"^(?:diagnosed|treated)\s+with\s+", None, None),
            (r"^(?:patient|pt|he|she|they)\s+(?:reports?|reported)\s+(?:of\s+)?", None, None),
            (r"^(?:patient|pt|he|she|they)\s+(?:complains?|complained)\s+of\s+", None, None),
            (r"^(?:patient|pt|he|she|they)\s+(?:has|have|had)\s+", None, None),
            # e.g., "Patient is a 65yo ... with HTN and DM2"
            (r"^(?:patient|pt|he|she|they)\b.*?\bwith\b\s+", None, None),
        ]

        for pat, assertion, subject in rules:
            m = re.match(pat, sentence, flags=re.IGNORECASE)
            if not m:
                continue

            prefix = sentence[: m.end()]
            # Avoid repeating extremely long prefixes.
            if len(prefix.strip()) <= 80:
                return prefix, assertion, subject

            # If the prefix is too long, skip stripping but keep any strong cues.
            return "", assertion, subject

        return "", None, None

    def _looks_like_template_scaffold(self, raw: str, candidate: str) -> bool:
        """
        Heuristic filter for templated field strings that should not become concepts.

        Keep this conservative: only drop spans that look like UI/section scaffolding
        (lab/result headers, field labels like "Types:", etc.).
        """
        raw_l = (raw or "").strip().lower()
        cand_l = (candidate or "").strip().lower()
        if not cand_l:
            return True

        if ("lab results" in cand_l or "laboratory results" in cand_l) and (
            self._lab_num_re.search(cand_l) or len(cand_l) <= 48
        ):
            return True

        # If field labels leak through, it's almost always templated noise.
        if "types:" in raw_l:
            return True
        if self._tail_field_label_re.search(candidate):
            return True

        return False

    def _split_mentions(self, sentence: str, prefix: str = "") -> List[str]:
        """Split a sentence into mention candidates (no clinical framing)."""
        return [m for m, _, _ in self._split_mentions_with_spans(sentence, prefix=prefix)]

    def _split_mentions_with_spans(self, sentence: str, prefix: str = "") -> List[Tuple[str, int, int]]:
        """
        Split a sentence into mention candidates and keep their spans.

        Returns:
            List of (mention_text, start, end) where start/end are indices into `sentence`.
        """
        sentence = sentence.strip()
        if len(sentence) <= 3:
            return []

        if not prefix:
            prefix, _, _ = self._detect_context(sentence)

        strip_chars = set(" \t\r\n-:|+")
        trim_chars = set(" \t\r\n-:()[]{}.,!?|+")

        content_start = len(prefix) if prefix else 0
        content_end = len(sentence)

        # Mirror the original `.strip()` behavior while keeping indices.
        while content_start < content_end and sentence[content_start] in strip_chars:
            content_start += 1
        while content_end > content_start and sentence[content_end - 1] in strip_chars:
            content_end -= 1

        if content_end - content_start <= 2:
            return []

        sep_re = re.compile(r"\s*(?:\t+|\s{3,}|,|;|/|~|\||\band\b|\bor\b)\s*", flags=re.IGNORECASE)
        spans: List[Tuple[int, int]] = []

        # Many clinical templates use repeated field headers mid-sentence, e.g.
        # "Day: 1 Types: Cigarettes Start date: ...". Split on these header boundaries
        # first so we can strip known headers and keep the useful values.
        seg_points = [content_start]
        # If the sentence itself starts with a header ("Daily Labs:", "Past Medical History:", ...),
        # avoid splitting inside that initial header phrase.
        initial_hdr_end = content_start
        m0 = self._header_prefix_re.match(sentence[content_start:content_end])
        if m0:
            hdr0 = self._normalize_header(m0.group(1))
            if hdr0 in self._section_headers:
                initial_hdr_end = content_start + int(m0.end())
        for m in self._mid_header_re.finditer(sentence, content_start, content_end):
            split_at = int(m.start(1))
            if content_start < split_at < content_end and split_at >= int(initial_hdr_end):
                seg_points.append(split_at)
        seg_points.append(content_end)
        seg_points = sorted(set(seg_points))

        for seg_s, seg_e in zip(seg_points, seg_points[1:]):
            last = int(seg_s)
            for m in sep_re.finditer(sentence, int(seg_s), int(seg_e)):
                spans.append((last, m.start()))
                last = m.end()
            spans.append((last, int(seg_e)))

        safe_2letter = getattr(self.abbrev_expander, "safe_2letter", set())
        out: List[Tuple[str, int, int]] = []
        for s, e in spans:
            # Trim punctuation/whitespace, keeping the accurate span.
            while s < e and sentence[s] in trim_chars:
                s += 1
            while e > s and sentence[e - 1] in trim_chars:
                e -= 1

            # Strip leading section-header prefixes like "Skin: ..." or "Breath sounds: ..."
            # while keeping offsets stable.
            while True:
                sub = sentence[s:e]
                m = self._header_prefix_re.match(sub)
                if not m:
                    break
                hdr = self._normalize_header(m.group(1))
                if hdr not in self._section_headers:
                    break
                s += int(m.end())
                while s < e and sentence[s] in trim_chars:
                    s += 1

            # Trim common low-signal prepositional fragments that show up due to list formatting.
            # This keeps offsets stable (we adjust s/e) and avoids spans like "of the abdomen" or
            # "vomiting for the".
            while True:
                sub = sentence[s:e]
                m = self._lead_stop_re.match(sub)
                if not m:
                    break
                s += int(m.end())
                while s < e and sentence[s] in trim_chars:
                    s += 1
            while True:
                sub = sentence[s:e]
                m = self._tail_incomplete_with_det_re.search(sub) or self._tail_incomplete_re.search(sub) or self._tail_determiner_re.search(sub)
                if not m:
                    break
                e = s + int(m.start())
                while e > s and sentence[e - 1] in trim_chars:
                    e -= 1

            # Strip trailing template field labels like "... Start date".
            while True:
                sub = sentence[s:e]
                m = self._tail_field_label_re.search(sub)
                if not m:
                    break
                e = s + int(m.start())
                while e > s and sentence[e - 1] in trim_chars:
                    e -= 1
            if e <= s:
                continue

            text = sentence[s:e]
            if len(text) > 2 or (len(text) == 2 and text.upper() in safe_2letter):
                out.append((text, s, e))

        if out:
            return out

        # Fallback: use the whole (trimmed) content span.
        whole = sentence[content_start:content_end]
        return [(whole, content_start, content_end)] if len(whole) > 2 else []
    
    def extract_single(
        self,
        text: str,
        threshold: float = 0.5,
        top_k: int = 10,
    ) -> Optional[ConceptExtraction]:
        """Extract concept from a single text span."""
        results = self.extract(text, threshold=threshold, extract_all=False, dedupe=True, top_k=top_k)
        return results[0] if results else None
    
    def extract(
        self,
        text: str,
        threshold: float = 0.5,
        extract_all: bool = True,
        use_model_labels: bool = False,
        dedupe: bool = True,
        top_k: int = 10,
        rerank: bool = True,
        lexical_weight: float = 0.30,
        rerank_margin: float = 0.04,
        relation_rerank: bool = False,
        relation_weight: float = 0.05,
        relation_max_degree: int = 2000,
        include_candidates: bool = False,
        clinical_rerank: bool = False,
        group_bias: Optional[Dict[str, float]] = None,
        tui_bias: Optional[Dict[str, float]] = None,
    ) -> List[ConceptExtraction]:
        """
        Extract medical concepts from clinical text.
        
        Args:
            text: Clinical text to process
            threshold: Minimum confidence score
            extract_all: If True, try to extract from each sentence/phrase
        
        Returns:
            List of extracted concepts
        """
        if not text or len(text.strip()) <= 3:
            return []

        sentence_spans: List[Tuple[int, int]] = []
        if extract_all:
            start = 0
            for m in re.finditer(r"[.;!\n]+", text):
                sentence_spans.append((start, m.start()))
                start = m.end()
            sentence_spans.append((start, len(text)))
        else:
            sentence_spans.append((0, len(text)))

        # Build sentence-level contexts + mention candidates (for concept matching).
        context_texts: List[str] = []
        context_overrides: List[Tuple[Optional[str], Optional[str]]] = []
        mention_raw: List[str] = []
        mention_expanded: List[str] = []
        mention_norm: List[str] = []
        mention_severity: List[Optional[str]] = []
        mention_laterality: List[Optional[str]] = []
        mention_temporality: List[Optional[str]] = []
        mention_sentence_idx: List[int] = []
        mention_start: List[int] = []
        mention_end: List[int] = []

        for sent_start, sent_end in sentence_spans:
            # Trim whitespace but keep stable offsets.
            while sent_start < sent_end and text[sent_start].isspace():
                sent_start += 1
            while sent_end > sent_start and text[sent_end - 1].isspace():
                sent_end -= 1

            if sent_end - sent_start <= 3:
                continue

            sent = text[sent_start:sent_end]
            prefix_raw, assertion_override, subject_override = self._detect_context(sent)
            expanded_sent = self.abbrev_expander.expand(sent)
            context_texts.append(expanded_sent)
            sidx = len(context_texts) - 1
            context_overrides.append((assertion_override, subject_override))

            raw_mentions = self._split_mentions_with_spans(sent, prefix=prefix_raw)
            if not raw_mentions:
                raw_mentions = [(sent, 0, len(sent))]

            # De-dupe within a sentence (case-insensitive) to avoid repeated work.
            seen = set()
            for raw, m_start, m_end in raw_mentions:
                exp = self.abbrev_expander.expand(raw)
                norm, sev, lat, tmp = self._extract_modifiers(exp)
                cand_text = (norm or exp).strip()
                cand_lower = cand_text.lower()
                if self._is_section_header(raw):
                    continue
                if not self._allow_short_span(raw):
                    continue
                if not self._has_alpha_re.search(cand_text):
                    continue
                if self._other_prefix_re.match(cand_text):
                    continue
                if cand_lower in self._stop_mentions:
                    continue
                if self._looks_like_template_scaffold(raw, cand_text):
                    continue
                key = exp.lower().strip()
                if key in seen:
                    continue
                seen.add(key)

                mention_raw.append(raw)
                mention_expanded.append(exp)
                mention_norm.append(cand_text)
                mention_severity.append(sev)
                mention_laterality.append(lat)
                mention_temporality.append(tmp)
                mention_sentence_idx.append(sidx)
                mention_start.append(sent_start + int(m_start))
                mention_end.append(sent_start + int(m_end))

        # Pick assertion + subject once per sentence context.
        model_sentence_labels: List[Tuple[str, str]]
        if use_model_labels:
            encoded_ctx = self.tokenizer(
                context_texts,
                max_length=self.model.config.max_seq_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                ctx_out = self.model(
                    encoded_ctx["input_ids"].to(self.device),
                    encoded_ctx["attention_mask"].to(self.device),
                    task="all",
                )

            ctx_assert = ctx_out["assertion_logits"].argmax(dim=-1).tolist()
            ctx_subj = ctx_out["subject_logits"].argmax(dim=-1).tolist()
            model_sentence_labels = [
                (ASSERTION_LABELS[a], SUBJECT_LABELS[s])
                for a, s in zip(ctx_assert, ctx_subj)
            ]
        else:
            model_sentence_labels = [("PRESENT", "PATIENT") for _ in context_texts]

        sentence_labels: List[Tuple[str, str]] = []
        for (model_assertion, model_subject), (ov_assertion, ov_subject) in zip(
            model_sentence_labels, context_overrides
        ):
            sentence_labels.append(
                (
                    ov_assertion or model_assertion,
                    ov_subject or model_subject,
                )
            )

        # Embed mention candidates in one batch and search the concept index.
        encoded_mentions = self.tokenizer(
            mention_norm,
            max_length=self.model.config.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            mention_out = self.model(
                encoded_mentions["input_ids"].to(self.device),
                encoded_mentions["attention_mask"].to(self.device),
                task="embedding",
            )

        emb_np = mention_out["embedding"].cpu().numpy().astype(np.float32)
        faiss.normalize_L2(emb_np)
        k = max(1, int(top_k))
        scores, indices = self.index.search(emb_np, k)

        # Optional semantic/type biases and prefterm penalties for clinical notes.
        # Keep this lightweight: it only reorders within FAISS top-k.
        group_bias_map: Dict[str, float] = dict(group_bias or {})
        tui_bias_map: Dict[str, float] = dict(tui_bias or {})
        use_penalties = bool(clinical_rerank)
        if clinical_rerank:
            # Merge defaults with user overrides (overrides win).
            merged_group = dict(self._default_group_bias)
            merged_group.update(group_bias_map)
            group_bias_map = merged_group

            merged_tui = dict(self._default_tui_bias)
            merged_tui.update({self._normalize_tui_key(k): float(v) for k, v in tui_bias_map.items()})
            tui_bias_map = merged_tui

        # Pass 1: pick a best candidate per mention with lightweight reranking.
        # Important: lexical/clinical/relation signals are used as tie-breakers only within a
        # small score band near the top FAISS hit, to avoid overriding the embedding similarity.
        best_choice: List[int] = []
        best_score_list: List[float] = []
        best_rerank_list: List[float] = []
        best_lex_list: List[float] = []
        base_best_list: List[float] = []
        cand_cache: Optional[List[List[Dict[str, Any]]]] = [] if include_candidates else None
        margin: Optional[float] = None
        try:
            margin_val = float(rerank_margin)
            margin = None if margin_val < 0 else float(margin_val)
        except Exception:
            margin = None

        for i in range(len(mention_expanded)):
            base_best = float(scores[i][0])
            base_best_list.append(base_best)
            if base_best < threshold:
                best_choice.append(0)
                best_score_list.append(base_best)
                best_rerank_list.append(base_best)
                best_lex_list.append(0.0)
                if cand_cache is not None:
                    cand_cache.append([])
                continue

            best_j = 0
            best_score = base_best
            best_rerank = base_best
            best_lex = 0.0

            cand_rows: List[Dict[str, Any]] = []
            compute_lex = bool(k > 1 and (rerank or relation_rerank or include_candidates))
            min_score = None
            if margin is not None and (rerank or relation_rerank or clinical_rerank):
                min_score = float(base_best) - float(margin)
            for j in range(k):
                cand_score = float(scores[i][j])
                cand_idx = int(indices[i][j])
                if cand_idx < 0:
                    continue
                eligible = True
                if min_score is not None and cand_score < min_score:
                    eligible = False
                cand_pref = self.prefterms[cand_idx]
                cand_cui_int = int(self.cuis[cand_idx])
                tui_str, _, cand_group, _ = self._get_tui_info(cand_cui_int)

                lex = self._jaccard(mention_expanded[i], cand_pref) if compute_lex else 0.0
                bias = float(group_bias_map.get(cand_group, 0.0)) + float(tui_bias_map.get(tui_str, 0.0))
                penalty = float(self._prefterm_penalty(str(cand_pref), span_text=mention_raw[i])) if use_penalties and float(lex) < 0.999 else 0.0

                cand_rerank = cand_score
                if k > 1 and rerank:
                    cand_rerank += float(lexical_weight) * float(lex)
                cand_rerank += bias + penalty
                if eligible and cand_score >= threshold and cand_rerank > best_rerank:
                    best_rerank = cand_rerank
                    best_score = cand_score
                    best_j = j
                if cand_cache is not None:
                    cand_rows.append(
                        {
                            "rank": int(j),
                            "cui": f"C{cand_cui_int:07d}",
                            "preferred_term": str(cand_pref),
                            "tui": tui_str,
                            "semantic_group": cand_group,
                            "score": float(cand_score),
                            "lex": float(lex),
                            "meets_threshold": bool(cand_score >= threshold),
                            "relation_hits": 0,
                            "bias": float(bias),
                            "penalty": float(penalty),
                            "rerank_score": float(cand_rerank),
                        }
                    )

            # Lex overlap for the chosen candidate (used to protect exact matches in relation rerank).
            if compute_lex:
                chosen_idx = int(indices[i][best_j])
                chosen_pref = self.prefterms[chosen_idx]
                best_lex = float(self._jaccard(mention_expanded[i], chosen_pref))

            best_choice.append(best_j)
            best_score_list.append(best_score)
            best_rerank_list.append(best_rerank)
            best_lex_list.append(best_lex)
            if cand_cache is not None:
                cand_cache.append(cand_rows)

        # Pass 2 (optional): relation-based rerank using MRREL neighbors in the same sentence.
        if relation_rerank:
            if self.mrrel_indptr is None or self.mrrel_indices is None:
                raise FileNotFoundError(
                    "relation_rerank requested but MRREL CSR not loaded. "
                    "Expected trm_umls/data/umls/mrrel_{indptr,indices}.npy (or pass paths to TRMUMLSPipeline.load)."
                )

            # Build sentence-level context sets from pass-1 picks.
            sent_to_rows: List[Set[int]] = [set() for _ in context_texts]
            for mi in range(len(mention_expanded)):
                if best_score_list[mi] < threshold:
                    continue
                chosen_idx = int(indices[mi][best_choice[mi]])
                sent_to_rows[mention_sentence_idx[mi]].add(chosen_idx)

            for mi in range(len(mention_expanded)):
                if best_score_list[mi] < threshold:
                    continue
                sidx = mention_sentence_idx[mi]
                base_choice = int(best_choice[mi])
                base_idx = int(indices[mi][base_choice])
                ctx = set(sent_to_rows[sidx])
                ctx.discard(base_idx)

                # Protect exact lexical matches: don't let relations override a perfect token-set match.
                protect = bool(best_lex_list[mi] >= 0.999)

                best_j = base_choice
                best_score = float(best_score_list[mi])
                best_rerank = float(best_rerank_list[mi])
                base_best = float(base_best_list[mi]) if mi < len(base_best_list) else float(scores[mi][0])
                min_score = None
                if margin is not None:
                    min_score = float(base_best) - float(margin)

                for j in range(k):
                    cand_score = float(scores[mi][j])
                    if cand_score < threshold:
                        break
                    if min_score is not None and cand_score < min_score:
                        break
                    cand_idx = int(indices[mi][j])
                    cand_pref = self.prefterms[cand_idx]
                    cand_cui_int = int(self.cuis[cand_idx])
                    tui_str, _, cand_group, _ = self._get_tui_info(cand_cui_int)
                    lex = self._jaccard(mention_expanded[mi], cand_pref) if (k > 1 and (rerank or include_candidates)) else 0.0
                    rel_hits = self._relation_hits(cand_idx, ctx, max_degree=int(relation_max_degree))
                    bias = float(group_bias_map.get(cand_group, 0.0)) + float(tui_bias_map.get(tui_str, 0.0))
                    penalty = float(self._prefterm_penalty(str(cand_pref), span_text=mention_raw[mi])) if use_penalties and float(lex) < 0.999 else 0.0
                    cand_rerank = cand_score
                    if k > 1 and rerank:
                        cand_rerank += float(lexical_weight) * float(lex)
                    cand_rerank += float(relation_weight) * float(rel_hits)
                    cand_rerank += bias + penalty

                    if cand_cache is not None and mi < len(cand_cache) and j < len(cand_cache[mi]):
                        cand_cache[mi][j]["relation_hits"] = int(rel_hits)
                        cand_cache[mi][j]["bias"] = float(bias)
                        cand_cache[mi][j]["penalty"] = float(penalty)
                        cand_cache[mi][j]["rerank_score"] = float(cand_rerank)

                    if protect and j != base_choice:
                        continue
                    if cand_rerank > best_rerank:
                        best_rerank = cand_rerank
                        best_score = cand_score
                        best_j = j

                best_choice[mi] = int(best_j)
                best_score_list[mi] = float(best_score)
                best_rerank_list[mi] = float(best_rerank)

        results: List[ConceptExtraction] = []
        for i in range(len(mention_expanded)):
            best_score = float(best_score_list[i])
            if best_score < threshold:
                continue

            best_j = int(best_choice[i])
            best_rerank = float(best_rerank_list[i])

            idx = int(indices[i][best_j])
            cui = int(self.cuis[idx])
            prefterm = self.prefterms[idx]
            tui, type_name, group, group_name = self._get_tui_info(cui)

            assertion, subject = sentence_labels[mention_sentence_idx[i]]

            results.append(
                ConceptExtraction(
                    text=mention_raw[i],
                    expanded_text=mention_expanded[i],
                    normalized_text=mention_norm[i],
                    cui=f"C{cui:07d}",
                    preferred_term=prefterm,
                    tui=tui,
                    semantic_type=type_name,
                    semantic_group=group,
                    semantic_group_name=group_name,
                    assertion=assertion,
                    subject=subject,
                    score=best_score,
                    rerank_score=best_rerank,
                    severity=mention_severity[i],
                    laterality=mention_laterality[i],
                    temporality=mention_temporality[i],
                    start=int(mention_start[i]),
                    end=int(mention_end[i]),
                    candidates=(cand_cache[i] if cand_cache is not None else None),
                )
            )

        if not dedupe:
            return sorted(results, key=lambda x: x.score, reverse=True)

        # De-duplicate by concept + context labels, keeping the best score.
        best: Dict[Tuple[str, str, str], ConceptExtraction] = {}
        for r in results:
            key = (r.cui, r.assertion, r.subject)
            if key not in best or r.score > best[key].score:
                best[key] = r

        return sorted(best.values(), key=lambda x: x.score, reverse=True)
    
    def to_dict(self, extractions: List[ConceptExtraction]) -> List[Dict]:
        """Convert extractions to list of dictionaries."""
        return [asdict(e) for e in extractions]
    
    def to_json(self, extractions: List[ConceptExtraction], indent: int = 2) -> str:
        """Convert extractions to JSON string."""
        return json.dumps(self.to_dict(extractions), indent=indent)


def main():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="TRM-UMLS concept extraction demo")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (defaults to V2 model, falling back to V1).",
    )
    parser.add_argument("--index-path", type=Path, default=None, help="FAISS index path (optional)")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Embedding metadata JSON (optional)")
    parser.add_argument("--tui-mappings-path", type=Path, default=None, help="CUI→TUI mappings JSON (optional)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=10, help="FAISS top-k candidates per mention")
    parser.add_argument("--no-rerank", action="store_true", help="Disable lexical reranking")
    parser.add_argument("--lexical-weight", type=float, default=0.30, help="Rerank weight for token overlap")
    parser.add_argument(
        "--rerank-margin",
        type=float,
        default=0.04,
        help="Only allow rerank to switch within this score delta of the top FAISS hit (set <0 to disable).",
    )
    parser.add_argument("--relation-rerank", action="store_true", help="Enable MRREL relation-based reranking (optional)")
    parser.add_argument("--relation-weight", type=float, default=0.05, help="Rerank weight per MRREL neighbor hit")
    parser.add_argument("--relation-max-degree", type=int, default=2000, help="Skip relation scoring for hub nodes above this degree")
    parser.add_argument("--clinical-rerank", action="store_true", help="Enable clinical rerank biases (semantic group/TUI + prefterm penalties)")
    parser.add_argument("--group-bias", type=str, default="", help="Comma-separated group=weight (e.g., DISO=0.01,PROC=0.005)")
    parser.add_argument("--tui-bias", type=str, default="", help="Comma-separated TUI=weight (e.g., T184=0.01,T033=-0.005)")
    parser.add_argument("--no-extract-all", action="store_true", help="Do not split into spans.")
    parser.add_argument("--no-dedupe", action="store_true", help="Keep all mention-level hits (no concept de-dupe).")
    parser.add_argument("--json", action="store_true", help="Output JSON for a single --text input.")
    parser.add_argument("--text", type=str, default=None, help="Run extraction on a single text string.")
    parser.add_argument(
        "--use-model-labels",
        action="store_true",
        help="Use model heads for assertion/subject when rules don't trigger (experimental).",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a small smoke test suite.")
    args = parser.parse_args()

    checkpoint_path: Path
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        model_pt = base_dir / "checkpoints" / "model.pt"
        model_last = base_dir / "checkpoints" / "model_last.pt"
        checkpoint_path = model_pt if model_pt.exists() else model_last

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    pipeline = TRMUMLSPipeline.load(
        str(checkpoint_path),
        index_path=str(args.index_path) if args.index_path is not None else None,
        metadata_path=str(args.metadata_path) if args.metadata_path is not None else None,
        tui_mappings_path=str(args.tui_mappings_path) if args.tui_mappings_path is not None else None,
        device=args.device,
    )

    if args.text is not None:
        results = pipeline.extract(
            args.text,
            threshold=args.threshold,
            extract_all=(not args.no_extract_all),
            use_model_labels=args.use_model_labels,
            dedupe=(not args.no_dedupe),
            top_k=int(args.top_k),
            rerank=(not bool(args.no_rerank)),
            lexical_weight=float(args.lexical_weight),
            rerank_margin=float(args.rerank_margin),
            relation_rerank=bool(args.relation_rerank),
            relation_weight=float(args.relation_weight),
            relation_max_degree=int(args.relation_max_degree),
            clinical_rerank=bool(args.clinical_rerank),
            group_bias=TRMUMLSPipeline._parse_bias_map(str(args.group_bias), kind="group"),
            tui_bias=TRMUMLSPipeline._parse_bias_map(str(args.tui_bias), kind="tui"),
        )
        if args.json:
            print(pipeline.to_json(results))
            return

        print(f"Input: {args.text!r}")
        for r in results:
            span = f"[{r.start}:{r.end}]" if r.start >= 0 and r.end >= 0 else "[-:-]"
            before_after = f"{r.text!r}→{r.expanded_text!r}" if r.text != r.expanded_text else repr(r.text)
            print(
                f"- {span} {before_after} | {r.cui} | {r.preferred_term} | {r.tui} | {r.assertion} | {r.subject} | {r.score:.3f}"
            )
        return

    if args.smoke:
        smoke_cases: List[Tuple[str, List[str]]] = [
            ("kidney stone", ["kidney"]),
            ("renal calculi", ["renal", "calculus"]),
            ("Patient denies HTN.", ["hypertension"]),
            ("Family history of DM and CAD.", ["diabetes", "coronary"]),
            ("No evidence of COPD.", ["pulmonary"]),
        ]

        passed = 0
        for text, expected_substrings in smoke_cases:
            extracted = pipeline.extract(
                text,
                threshold=args.threshold,
                extract_all=True,
                use_model_labels=args.use_model_labels,
                dedupe=True,
                top_k=int(args.top_k),
                rerank=(not bool(args.no_rerank)),
                lexical_weight=float(args.lexical_weight),
                rerank_margin=float(args.rerank_margin),
            )
            terms = [e.preferred_term.lower() for e in extracted]
            ok = all(any(exp.lower() in t for t in terms) for exp in expected_substrings)
            passed += 1 if ok else 0
            status = "✓" if ok else "✗"
            print(f"{status} {text!r}")
            for e in extracted[:3]:
                print(f"  - {e.cui} | {e.preferred_term} | {e.assertion} | {e.subject} | {e.score:.3f}")

        print(f"\nSmoke: {passed}/{len(smoke_cases)} passed")
        return

    test_texts = [
        "Patient denies HTN.",
        "Family history of DM and CAD.",
        "No evidence of COPD.",
        "Diagnosed with pneumonia.",
        "Possible pulmonary embolism.",
        "Patient has hypertension and diabetes mellitus.",
    ]

    print("\n" + "=" * 70)
    print("CONCEPT EXTRACTION RESULTS")
    print("=" * 70)

    for text in test_texts:
        print(f"\nInput: '{text}'")
        print("-" * 50)

        results = pipeline.extract(
            text,
            threshold=args.threshold,
            dedupe=(not args.no_dedupe),
            top_k=int(args.top_k),
            rerank=(not bool(args.no_rerank)),
            lexical_weight=float(args.lexical_weight),
            rerank_margin=float(args.rerank_margin),
        )

        if not results:
            print("  No concepts extracted")
            continue

        for r in results:
            print(f"  CUI: {r.cui}")
            print(f"  Term: {r.preferred_term}")
            print(f"  TUI: {r.tui} ({r.semantic_type})")
            print(f"  Group: {r.semantic_group} ({r.semantic_group_name})")
            print(f"  Assertion: {r.assertion}")
            print(f"  Subject: {r.subject}")
            print(f"  Score: {r.score:.3f}")
            if r.text != r.expanded_text:
                print(f"  Expanded: {r.text} → {r.expanded_text}")


if __name__ == "__main__":
    main()
