export type Assertion = "PRESENT" | "ABSENT" | "POSSIBLE";

export type SemanticGroup =
  | "ACTI"
  | "ANAT"
  | "CHEM"
  | "CONC"
  | "DEVI"
  | "DISO"
  | "GENE"
  | "GEOG"
  | "LIVB"
  | "OBJC"
  | "OCCU"
  | "ORGA"
  | "PHEN"
  | "PHYS"
  | "PROC";

export interface ConceptExtraction {
  text: string;
  expanded_text: string;
  normalized_text: string;
  cui: string;
  preferred_term: string;
  tui: string;
  semantic_type: string;
  semantic_group: string;
  semantic_group_name: string;
  assertion: Assertion;
  subject: "PATIENT" | "FAMILY" | "OTHER";
  score: number;
  rerank_score: number;
  severity: string | null;
  laterality: string | null;
  temporality: string | null;
  start: number;
  end: number;
  candidates: unknown[] | null;
}

export interface ExtractOptions {
  threshold: number;
  top_k: number;
  dedupe: boolean;
  clinical_rerank: boolean;
  rerank: boolean;
  include_candidates: boolean;
  relation_rerank: boolean;
}

export interface ExtractResponse {
  extractions: ConceptExtraction[];
  meta: Record<string, unknown>;
}

export interface NoteIn {
  id: string;
  name: string;
  text: string;
}

export interface ExtractBatchItem {
  id: string;
  name: string;
  extractions: ConceptExtraction[];
  meta: Record<string, unknown>;
}

export interface ExtractBatchResponse {
  results: ExtractBatchItem[];
  meta: Record<string, unknown>;
}

export interface NoteResult {
  id: string;
  name: string;
  text: string;
  extractions: ConceptExtraction[];
  meta: Record<string, unknown>;
}
