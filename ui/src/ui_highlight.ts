import type { ConceptExtraction } from "./types";

export type HighlightSegment =
  | { text: string; extraction?: undefined }
  | { text: string; extraction: ConceptExtraction };

export function buildSegmentsNonOverlapping(text: string, extractions: ConceptExtraction[]): HighlightSegment[] {
  const spans = extractions
    .filter((e) => Number.isFinite(e.start) && Number.isFinite(e.end))
    .filter((e) => e.start >= 0 && e.end > e.start && e.end <= text.length)
    .sort((a, b) => a.start - b.start || b.end - a.end);

  const out: HighlightSegment[] = [];
  let idx = 0;
  for (const e of spans) {
    if (e.start < idx) continue; // overlap with already-chosen span
    if (idx < e.start) out.push({ text: text.slice(idx, e.start) });
    out.push({ text: text.slice(e.start, e.end), extraction: e });
    idx = e.end;
  }
  if (idx < text.length) out.push({ text: text.slice(idx) });
  return out.length ? out : [{ text }];
}

