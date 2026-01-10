import { useMemo, useRef, useState } from "react";
import { extractBatch, extractSingle } from "./api";
import type {
  Assertion,
  ConceptExtraction,
  ExtractOptions,
  NoteResult,
  SemanticGroup,
} from "./types";
import { assertionBadgeClass, groupBadgeClass, groupMarkClass } from "./ui_colors";
import { buildSegmentsNonOverlapping } from "./ui_highlight";

const DEFAULT_OPTIONS: ExtractOptions = {
  threshold: 0.55,
  top_k: 10,
  dedupe: true,
  clinical_rerank: true,
  rerank: true,
  include_candidates: false,
  relation_rerank: false,
};

function App() {
  const [options, setOptions] = useState<ExtractOptions>(DEFAULT_OPTIONS);
  const [text, setText] = useState<string>(
    "Assessment: HTN, DM, COPD. Severe left knee pain.\nDenies chest pain or shortness of breath at rest.\nFamily history: breast cancer in mother.",
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [results, setResults] = useState<NoteResult[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  const [assertionFilter, setAssertionFilter] = useState<Record<Assertion, boolean>>({
    PRESENT: true,
    ABSENT: true,
    POSSIBLE: true,
  });
  const [groupFilter, setGroupFilter] = useState<Record<string, boolean>>({});
  const [query, setQuery] = useState("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const active = useMemo(() => results.find((r) => r.id === activeId) ?? null, [results, activeId]);

  const availableGroups = useMemo(() => {
    const out = new Set<string>();
    for (const r of results) {
      for (const e of r.extractions) out.add(e.semantic_group);
    }
    return Array.from(out).sort();
  }, [results]);

  const availableAssertions = useMemo(() => {
    const out = new Set<Assertion>();
    for (const r of results) {
      for (const e of r.extractions) out.add(e.assertion);
    }
    return Array.from(out).sort();
  }, [results]);

  const filteredExtractions = useMemo(() => {
    if (!active) return [];
    const q = query.trim().toLowerCase();
    return active.extractions.filter((e) => {
      if (!assertionFilter[e.assertion]) return false;
      if (groupFilter[e.semantic_group] === false) return false;
      if (!q) return true;
      return (
        e.text.toLowerCase().includes(q) ||
        e.preferred_term.toLowerCase().includes(q) ||
        e.cui.toLowerCase().includes(q)
      );
    });
  }, [active, assertionFilter, groupFilter, query]);

  const counts = useMemo(() => {
    const byAssertion: Record<string, number> = {};
    const byGroup: Record<string, number> = {};
    for (const e of filteredExtractions) {
      byAssertion[e.assertion] = (byAssertion[e.assertion] ?? 0) + 1;
      byGroup[e.semantic_group] = (byGroup[e.semantic_group] ?? 0) + 1;
    }
    return { byAssertion, byGroup };
  }, [filteredExtractions]);

  const selected = useMemo(() => {
    if (!active || !selectedKey) return null;
    return active.extractions.find((e) => extractionKey(e) === selectedKey) ?? null;
  }, [active, selectedKey]);

  const segments = useMemo(() => {
    if (!active) return [];
    return buildSegmentsNonOverlapping(active.text, filteredExtractions);
  }, [active, filteredExtractions]);

  async function runFromText() {
    setBusy(true);
    setError(null);
    setSelectedKey(null);
    try {
      const id = crypto.randomUUID();
      const res = await extractSingle(text, options);
      const next: NoteResult = {
        id,
        name: "pasted text",
        text,
        extractions: res.extractions,
        meta: res.meta,
      };
      setResults([next]);
      setActiveId(id);
      setGroupFilter((prev) => hydrateGroupFilter(prev, next.extractions));
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  async function runFromFiles(files: FileList) {
    setBusy(true);
    setError(null);
    setSelectedKey(null);
    try {
      const notes: NoteResult[] = [];
      for (const file of Array.from(files)) {
        const id = crypto.randomUUID();
        const fileText = await file.text();
        notes.push({ id, name: file.name, text: fileText, extractions: [], meta: {} });
      }

      const batchRes = await extractBatch(
        notes.map((n) => ({ id: n.id, name: n.name, text: n.text })),
        options,
      );

      const merged: NoteResult[] = notes.map((n) => {
        const hit = batchRes.results.find((r) => r.id === n.id);
        return {
          ...n,
          extractions: hit?.extractions ?? [],
          meta: hit?.meta ?? {},
        };
      });

      setResults(merged);
      setActiveId(merged[0]?.id ?? null);
      setGroupFilter((prev) => hydrateGroupFilter(prev, merged.flatMap((m) => m.extractions)));
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-full">
      <header className="border-b border-slate-800 bg-slate-950/60 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
          <div className="space-y-1">
            <div className="text-lg font-semibold tracking-tight">trm-umls viewer</div>
            <div className="text-sm text-slate-400">
              paste a note or upload files. results stay local.
            </div>
          </div>
          <a className="text-sm" href="https://github.com/editnori" target="_blank" rel="noreferrer">
            github
          </a>
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-6 px-4 py-6 lg:grid-cols-2">
        <section className="space-y-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold text-slate-200">input</div>
              <div className="text-xs text-slate-400">
                api: {import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000"}
              </div>
            </div>

            <div className="mt-3 space-y-3">
              <textarea
                className="h-56 w-full resize-none rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 font-mono text-sm leading-6 text-slate-100 outline-none focus:border-slate-600"
                value={text}
                onChange={(e) => setText(e.target.value)}
                spellCheck={false}
              />

              <div className="flex flex-wrap items-center gap-3">
                <label className="text-sm text-slate-300">
                  threshold
                  <input
                    className="ml-2 w-24 rounded-md border border-slate-800 bg-slate-950/60 px-2 py-1 text-sm"
                    type="number"
                    step="0.01"
                    min={0}
                    max={1}
                    value={options.threshold}
                    onChange={(e) => setOptions((o) => ({ ...o, threshold: clamp01(e.target.value) }))}
                  />
                </label>

                <label className="text-sm text-slate-300">
                  top-k
                  <input
                    className="ml-2 w-20 rounded-md border border-slate-800 bg-slate-950/60 px-2 py-1 text-sm"
                    type="number"
                    step="1"
                    min={1}
                    max={50}
                    value={options.top_k}
                    onChange={(e) => setOptions((o) => ({ ...o, top_k: clampInt(e.target.value, 1, 50) }))}
                  />
                </label>

                <Toggle
                  label="clinical rerank"
                  checked={options.clinical_rerank}
                  onChange={(v) => setOptions((o) => ({ ...o, clinical_rerank: v }))}
                />
                <Toggle label="dedupe" checked={options.dedupe} onChange={(v) => setOptions((o) => ({ ...o, dedupe: v }))} />
              </div>

              <div className="flex flex-wrap gap-3">
                <button
                  className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-semibold text-white hover:bg-sky-500 disabled:opacity-50"
                  onClick={runFromText}
                  disabled={busy || text.trim().length < 1}
                >
                  {busy ? "running..." : "run on pasted text"}
                </button>

                <button
                  className="rounded-lg border border-slate-700 bg-slate-950/40 px-4 py-2 text-sm font-semibold text-slate-100 hover:bg-slate-950/70 disabled:opacity-50"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={busy}
                >
                  run on uploaded files
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    const files = e.target.files;
                    if (files && files.length > 0) void runFromFiles(files);
                    e.target.value = "";
                  }}
                />
              </div>

              {error ? (
                <div className="rounded-lg border border-rose-900/50 bg-rose-950/40 px-3 py-2 text-sm text-rose-200">
                  {error}
                </div>
              ) : null}
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold text-slate-200">notes</div>
              <div className="text-xs text-slate-400">{results.length ? `${results.length} loaded` : "none loaded"}</div>
            </div>

            <div className="mt-3 space-y-2">
              {results.length ? (
                results.map((r) => (
                  <button
                    key={r.id}
                    className={[
                      "w-full rounded-lg border px-3 py-2 text-left text-sm",
                      r.id === activeId
                        ? "border-sky-500/60 bg-sky-500/10 text-slate-50"
                        : "border-slate-800 bg-slate-950/40 text-slate-200 hover:bg-slate-950/70",
                    ].join(" ")}
                    onClick={() => {
                      setActiveId(r.id);
                      setSelectedKey(null);
                      setGroupFilter((prev) => hydrateGroupFilter(prev, r.extractions));
                    }}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="truncate">{r.name}</div>
                      <div className="shrink-0 text-xs text-slate-400">
                        {typeof r.meta?.count === "number" ? `${r.meta.count} rows` : ""}
                      </div>
                    </div>
                  </button>
                ))
              ) : (
                <div className="text-sm text-slate-400">run on text or upload notes to see results.</div>
              )}
            </div>
          </div>
        </section>

        <section className="space-y-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="text-sm font-semibold text-slate-200">results</div>
              <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
                {active ? (
                  <>
                    <span>{active.name}</span>
                    {typeof active.meta?.ms === "number" ? <span>· {active.meta.ms} ms</span> : null}
                    {typeof active.meta?.count === "number" ? <span>· {active.meta.count} rows</span> : null}
                  </>
                ) : (
                  <span>no note selected</span>
                )}
              </div>
            </div>

            <div className="mt-3 grid gap-3 md:grid-cols-2">
              <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
                <div className="text-xs font-semibold text-slate-300">filters</div>
                <div className="mt-2 space-y-3">
                  <div className="flex flex-wrap items-center gap-2">
                    {availableAssertions.map((a) => (
                      <FilterChip
                        key={a}
                        active={assertionFilter[a]}
                        label={`${a} (${counts.byAssertion[a] ?? 0})`}
                        className={assertionBadgeClass(a)}
                        onClick={() =>
                          setAssertionFilter((prev) => ({ ...prev, [a]: !prev[a] }))
                        }
                      />
                    ))}
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    {availableGroups.map((g) => (
                      <FilterChip
                        key={g}
                        active={groupFilter[g] !== false}
                        label={`${g} (${counts.byGroup[g] ?? 0})`}
                        className={groupBadgeClass(g as SemanticGroup)}
                        onClick={() => setGroupFilter((prev) => ({ ...prev, [g]: prev[g] === false }))}
                      />
                    ))}
                  </div>

                  <input
                    className="w-full rounded-md border border-slate-800 bg-slate-950/60 px-2 py-1 text-sm text-slate-100 outline-none focus:border-slate-600"
                    placeholder="search span / term / cui"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </div>
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
                <div className="text-xs font-semibold text-slate-300">selection</div>
                <div className="mt-2 text-sm text-slate-200">
                  {selected ? (
                    <div className="space-y-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className={groupBadgeClass(selected.semantic_group as SemanticGroup)}>
                          {selected.semantic_group}
                        </span>
                        <span className={assertionBadgeClass(selected.assertion)}>{selected.assertion}</span>
                        <span className="rounded-md border border-slate-800 bg-slate-950/60 px-2 py-1 text-xs text-slate-300">
                          {selected.subject}
                        </span>
                      </div>
                      <div className="font-mono text-xs text-slate-300">
                        {selected.start}–{selected.end}
                      </div>
                      <div className="text-sm font-semibold">{selected.text}</div>
                      <div className="text-sm text-slate-300">{selected.preferred_term}</div>
                      <div className="text-sm text-slate-300">
                        <span className="font-mono">{selected.cui}</span>{" "}
                        <span className="text-slate-500">·</span> score {selected.score.toFixed(3)}
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-slate-400">
                      click a highlighted span or table row to see details.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
            <div className="text-sm font-semibold text-slate-200">highlight</div>
            <div className="mt-3 h-72 overflow-auto rounded-lg border border-slate-800 bg-slate-950/60 p-3 font-mono text-sm leading-6 text-slate-100">
              {active ? (
                <pre className="whitespace-pre-wrap">
                  {segments.map((seg, idx) => {
                    if (!seg.extraction) return <span key={idx}>{seg.text}</span>;
                    const e = seg.extraction;
                    const key = extractionKey(e);
                    const selected = selectedKey === key;
                    return (
                      <mark
                        key={idx}
                        className={[
                          "rounded px-1 py-0.5",
                          groupMarkClass(e.semantic_group as SemanticGroup),
                          selected ? "ring-2 ring-white/70" : "",
                        ].join(" ")}
                        onClick={() => setSelectedKey(key)}
                        title={`${e.preferred_term} (${e.cui})`}
                      >
                        {seg.text}
                        <span className="ml-1 align-top text-[10px] font-semibold uppercase tracking-wide text-white/90">
                          <span className={["rounded px-1 py-0.5", groupBadgeClass(e.semantic_group as SemanticGroup)].join(" ")}>
                            {e.semantic_group}
                          </span>
                          <span className={["ml-1 rounded px-1 py-0.5", assertionBadgeClass(e.assertion)].join(" ")}>
                            {e.assertion}
                          </span>
                        </span>
                      </mark>
                    );
                  })}
                </pre>
              ) : (
                <div className="text-sm text-slate-400">no note selected.</div>
              )}
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold text-slate-200">table</div>
              <div className="text-xs text-slate-400">{active ? `${filteredExtractions.length} shown` : ""}</div>
            </div>
            <div className="mt-3 h-72 overflow-auto rounded-lg border border-slate-800 bg-slate-950/60">
              <table className="w-full border-collapse text-left text-sm">
                <thead className="sticky top-0 bg-slate-950/90 text-xs text-slate-300">
                  <tr>
                    <th className="px-3 py-2">span</th>
                    <th className="px-3 py-2">cui</th>
                    <th className="px-3 py-2">term</th>
                    <th className="px-3 py-2">score</th>
                    <th className="px-3 py-2">assertion</th>
                    <th className="px-3 py-2">group</th>
                  </tr>
                </thead>
                <tbody className="text-slate-100">
                  {active ? (
                    filteredExtractions.map((e) => {
                      const key = extractionKey(e);
                      const selected = selectedKey === key;
                      return (
                        <tr
                          key={key}
                          className={[
                            "cursor-pointer border-t border-slate-900/60 hover:bg-slate-900/40",
                            selected ? "bg-slate-900/60" : "",
                          ].join(" ")}
                          onClick={() => setSelectedKey(key)}
                        >
                          <td className="px-3 py-2 font-mono text-xs text-slate-200">{e.text}</td>
                          <td className="px-3 py-2 font-mono text-xs text-slate-300">{e.cui}</td>
                          <td className="px-3 py-2 text-slate-200">{e.preferred_term}</td>
                          <td className="px-3 py-2 font-mono text-xs text-slate-300">{e.score.toFixed(3)}</td>
                          <td className="px-3 py-2">
                            <span className={assertionBadgeClass(e.assertion)}>{e.assertion}</span>
                          </td>
                          <td className="px-3 py-2">
                            <span className={groupBadgeClass(e.semantic_group as SemanticGroup)}>{e.semantic_group}</span>
                          </td>
                        </tr>
                      );
                    })
                  ) : (
                    <tr>
                      <td className="px-3 py-3 text-slate-400" colSpan={6}>
                        no note selected.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function clamp01(v: string): number {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0.55;
  return Math.max(0, Math.min(1, n));
}

function clampInt(v: string, lo: number, hi: number): number {
  const n = Math.floor(Number(v));
  if (!Number.isFinite(n)) return lo;
  return Math.max(lo, Math.min(hi, n));
}

function extractionKey(e: ConceptExtraction): string {
  return `${e.start}:${e.end}:${e.cui}:${e.assertion}:${e.subject}`;
}

function hydrateGroupFilter(prev: Record<string, boolean>, extractions: ConceptExtraction[]): Record<string, boolean> {
  const out = { ...prev };
  for (const e of extractions) {
    if (!(e.semantic_group in out)) out[e.semantic_group] = true;
  }
  return out;
}

function Toggle(props: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 text-sm text-slate-300">
      <input
        type="checkbox"
        checked={props.checked}
        onChange={(e) => props.onChange(e.target.checked)}
        className="h-4 w-4 rounded border-slate-700 bg-slate-950 text-sky-500"
      />
      {props.label}
    </label>
  );
}

function FilterChip(props: { active: boolean; label: string; className: string; onClick: () => void }) {
  return (
    <button
      className={[
        "rounded-md border px-2 py-1 text-xs font-semibold",
        props.active ? "border-white/10" : "border-slate-800 opacity-50",
        props.className,
      ].join(" ")}
      onClick={props.onClick}
      type="button"
    >
      {props.label}
    </button>
  );
}

export default App;
