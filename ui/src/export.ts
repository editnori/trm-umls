import * as XLSX from "xlsx";
import type { ConceptExtraction, NoteResult } from "./types";

type ExportRow = {
  note: string;
  start: number;
  end: number;
  span: string;
  cui: string;
  preferred_term: string;
  score: number;
  assertion: string;
  semantic_group: string;
  subject: string;
  tui: string;
  semantic_type: string;
  severity: string;
  laterality: string;
  temporality: string;
};

const EXPORT_HEADER: Array<keyof ExportRow> = [
  "note",
  "start",
  "end",
  "span",
  "cui",
  "preferred_term",
  "score",
  "assertion",
  "semantic_group",
  "subject",
  "tui",
  "semantic_type",
  "severity",
  "laterality",
  "temporality",
];

function safe(v: unknown): string {
  if (v === null || v === undefined) return "";
  return String(v);
}

function toRow(noteName: string, e: ConceptExtraction): ExportRow {
  return {
    note: noteName,
    start: e.start,
    end: e.end,
    span: e.text,
    cui: e.cui,
    preferred_term: e.preferred_term,
    score: Number.isFinite(e.score) ? e.score : 0,
    assertion: e.assertion,
    semantic_group: e.semantic_group,
    subject: e.subject,
    tui: e.tui,
    semantic_type: e.semantic_type,
    severity: safe(e.severity),
    laterality: safe(e.laterality),
    temporality: safe(e.temporality),
  };
}

function downloadBytes(filename: string, bytes: ArrayBuffer, mime: string): void {
  const blob = new Blob([bytes], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadCsv(results: NoteResult[], filename = "trm_umls_extractions.csv"): void {
  const rows: ExportRow[] = [];
  for (const r of results) for (const e of r.extractions) rows.push(toRow(r.name, e));
  const header = EXPORT_HEADER;
  const lines = [
    header.join(","),
    ...rows.map((row) =>
      header
        .map((k) => {
          const v = row[k];
          const s = typeof v === "number" ? String(v) : String(v ?? "");
          const escaped = s.replaceAll('"', '""');
          return `"${escaped}"`;
        })
        .join(","),
    ),
  ];
  downloadBytes(filename, new TextEncoder().encode(lines.join("\n")).buffer, "text/csv;charset=utf-8");
}

export function downloadXlsx(results: NoteResult[], filename = "trm_umls_extractions.xlsx"): void {
  const rows: ExportRow[] = [];
  for (const r of results) for (const e of r.extractions) rows.push(toRow(r.name, e));

  const header = EXPORT_HEADER.map((h) => String(h));
  const ws = XLSX.utils.json_to_sheet(rows, { header });
  if (rows.length === 0) XLSX.utils.sheet_add_aoa(ws, [header], { origin: "A1" });
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "extractions");
  const bytes = XLSX.write(wb, { bookType: "xlsx", type: "array" }) as ArrayBuffer;
  downloadBytes(filename, bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
}
