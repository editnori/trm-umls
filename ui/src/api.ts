import type { ExtractBatchResponse, ExtractOptions, ExtractResponse, NoteIn } from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8000";

export type HealthResponse = { ok: boolean; loaded: boolean };

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`health ${res.status}: ${await res.text()}`);
  return (await res.json()) as HealthResponse;
}

async function jsonFetch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`api ${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as T;
}

export async function extractSingle(text: string, options: ExtractOptions): Promise<ExtractResponse> {
  return await jsonFetch<ExtractResponse>("/extract", { text, options });
}

export async function extractBatch(notes: NoteIn[], options: ExtractOptions): Promise<ExtractBatchResponse> {
  return await jsonFetch<ExtractBatchResponse>("/extract_batch", { notes, options });
}
