import { CloudArrowUp, Copy, DownloadSimple, FileText, Play, Pulse, WarningCircle } from "@phosphor-icons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { extractBatch, extractSingle, getHealth } from "./api";
import { cx } from "./components/cx";
import { Card, Button, Pill, Stepper, Switch, Textarea } from "./components/ui";
import { downloadCsv, downloadXlsx } from "./export";
import type { Assertion, ConceptExtraction, ExtractOptions, NoteResult, SemanticGroup } from "./types";
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

export default function App() {
  const [options, setOptions] = useState<ExtractOptions>(DEFAULT_OPTIONS);
  const [text, setText] = useState<string>(
    "Assessment: HTN, DM, COPD. Severe left knee pain.\nDenies chest pain or shortness of breath at rest.\nFamily history: breast cancer in mother.",
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiOk, setApiOk] = useState<null | { ok: boolean; loaded: boolean }>(null);

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
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const active = useMemo(() => results.find((r) => r.id === activeId) ?? null, [results, activeId]);

  const availableGroups = useMemo(() => {
    const out = new Set<string>();
    for (const r of results) for (const e of r.extractions) out.add(e.semantic_group);
    return Array.from(out).sort();
  }, [results]);

  const availableAssertions = useMemo(() => {
    const out = new Set<Assertion>();
    for (const r of results) for (const e of r.extractions) out.add(e.assertion);
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

  useEffect(() => {
    let mounted = true;
    async function tick() {
      try {
        const h = await getHealth();
        if (mounted) setApiOk(h);
      } catch {
        if (mounted) setApiOk({ ok: false, loaded: false });
      }
    }
    void tick();
    const id = window.setInterval(tick, 5000);
    return () => {
      mounted = false;
      window.clearInterval(id);
    };
  }, []);

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
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
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
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-20 border-b border-white/5 bg-slate-950/65 backdrop-blur">
        <div className="mx-auto flex max-w-[1200px] items-center justify-between px-4 py-3">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-slate-950/40">
              <Pulse size={18} className="text-sky-300" />
            </div>
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold tracking-tight text-slate-100">trm-umls</div>
              <div className="truncate text-xs text-slate-400">local concept extraction + cui linking</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <StatusPill apiOk={apiOk} />
            <a
              className="text-sm text-slate-300 hover:text-slate-100"
              href="https://github.com/editnori/trm-umls"
              target="_blank"
              rel="noreferrer"
            >
              docs
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1200px] gap-4 px-4 py-4 lg:grid-cols-[360px_1fr] xl:grid-cols-[360px_1fr_420px]">
        <aside className="space-y-4">
          <Card
            title="input"
            subtitle="paste a note or drop files. nothing is uploaded anywhere."
            right={
              <div className="text-xs text-slate-500">
                api{" "}
                <span className="font-mono tabular-nums text-slate-400">
                  {(import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8000"}
                </span>
              </div>
            }
          >
            <div className="space-y-3">
              <Textarea
                ref={textareaRef}
                value={text}
                onChange={setText}
                rows={12}
                className="resize-none"
                placeholder="paste clinical text here…"
                onKeyDown={(e) => {
                  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") void runFromText();
                }}
              />

              <div className="grid gap-2">
                <div className="grid gap-2 sm:grid-cols-2">
                  <Stepper
                    label="threshold"
                    value={options.threshold}
                    onChange={(v) => setOptions((o) => ({ ...o, threshold: round2(clamp(v, 0, 1)) }))}
                    step={0.01}
                    min={0}
                    max={1}
                    format={(v) => v.toFixed(2)}
                  />
                  <Stepper
                    label="top-k"
                    value={options.top_k}
                    onChange={(v) => setOptions((o) => ({ ...o, top_k: Math.round(clamp(v, 1, 50)) }))}
                    step={1}
                    min={1}
                    max={50}
                  />
                </div>
                <Switch
                  label="clinical rerank"
                  description="adds lightweight clinical heuristics after retrieval"
                  checked={options.clinical_rerank}
                  onChange={(v) => setOptions((o) => ({ ...o, clinical_rerank: v }))}
                />
                <div className="grid gap-2 sm:grid-cols-2">
                  <Switch
                    label="dedupe"
                    description="drop overlapping duplicates"
                    checked={options.dedupe}
                    onChange={(v) => setOptions((o) => ({ ...o, dedupe: v }))}
                  />
                  <Switch
                    label="include candidates"
                    description="debug: include candidate list"
                    checked={options.include_candidates}
                    onChange={(v) => setOptions((o) => ({ ...o, include_candidates: v }))}
                  />
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant="primary"
                  disabled={busy || text.trim().length < 1 || apiOk?.ok === false}
                  onClick={runFromText}
                  className="min-w-[168px]"
                >
                  {busy ? <Pulse size={16} className="animate-pulse" /> : <Play size={16} />}
                  run (⌘/ctrl+enter)
                </Button>

                <Button
                  variant="secondary"
                  disabled={busy || apiOk?.ok === false}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <CloudArrowUp size={16} />
                  upload
                </Button>

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

                <Button
                  variant="ghost"
                  disabled={busy}
                  onClick={() => {
                    setText("");
                    textareaRef.current?.focus();
                  }}
                >
                  clear
                </Button>
              </div>

              {apiOk?.ok === false ? (
                <div className="rounded-lg border border-amber-400/20 bg-amber-600/10 px-3 py-2 text-xs text-amber-100">
                  api not reachable. start it with{" "}
                  <span className="font-mono text-amber-100/90">python3 -m trm_umls.api</span>.
                </div>
              ) : null}

              {error ? (
                <div className="rounded-lg border border-rose-900/40 bg-rose-950/35 px-3 py-2 text-sm text-rose-200">
                  <div className="flex items-start gap-2">
                    <WarningCircle size={18} className="mt-0.5 shrink-0 text-rose-300" />
                    <div className="min-w-0">
                      <div className="text-xs font-semibold uppercase tracking-wide text-rose-200">error</div>
                      <div className="mt-1 break-words font-mono text-xs text-rose-100">{error}</div>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </Card>

          <Card
            title="notes"
            subtitle={results.length ? `${results.length} loaded` : "none loaded"}
            right={
              <Button
                variant="ghost"
                disabled={!results.length}
                onClick={() => {
                  setResults([]);
                  setActiveId(null);
                  setSelectedKey(null);
                  setQuery("");
                  setGroupFilter({});
                }}
              >
                clear
              </Button>
            }
          >
            <div
              className="rounded-lg border border-dashed border-white/10 bg-slate-950/25 p-3"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (busy || apiOk?.ok === false) return;
                const files = e.dataTransfer.files;
                if (files && files.length) void runFromFiles(files);
              }}
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-slate-950/45">
                  <FileText size={16} className="text-slate-200" />
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-slate-200">drop files here</div>
                  <div className="mt-0.5 text-xs text-slate-400">or click upload above</div>
                </div>
              </div>
            </div>

            <div className="mt-3 space-y-2">
              {results.length ? (
                results.map((r) => (
                  <button
                    key={r.id}
                    className={cx(
                      "w-full rounded-lg border px-3 py-2 text-left text-sm transition-colors",
                      "border-white/10 bg-slate-950/35 hover:bg-slate-950/55",
                      r.id === activeId ? "ring-2 ring-sky-400/25" : "",
                    )}
                    onClick={() => {
                      setActiveId(r.id);
                      setSelectedKey(null);
                      setGroupFilter((prev) => hydrateGroupFilter(prev, r.extractions));
                    }}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="truncate text-slate-200">{r.name}</div>
                      <div className="shrink-0 font-mono text-xs tabular-nums text-slate-400">
                        {typeof r.meta?.count === "number" ? r.meta.count : "—"}
                      </div>
                    </div>
                    <div className="mt-1 flex items-center justify-between text-xs text-slate-500">
                      <span className="truncate">{r.text.slice(0, 80).replaceAll(/\s+/g, " ")}</span>
                      <span className="shrink-0">{typeof r.meta?.ms === "number" ? `${r.meta.ms} ms` : ""}</span>
                    </div>
                  </button>
                ))
              ) : (
                <div className="text-sm text-slate-400">run on text or drop files to see results.</div>
              )}
            </div>
          </Card>
        </aside>

        <section className="space-y-4">
          <Card
            title="document"
            subtitle={active ? active.name : "no note selected"}
            right={
              active ? (
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <Pill title="runtime (ms)">
                    <span className="font-mono tabular-nums">{typeof active.meta?.ms === "number" ? active.meta.ms : "—"}</span>
                    <span className="ml-1 text-slate-500">ms</span>
                  </Pill>
                  <Pill title="extractions">
                    <span className="font-mono tabular-nums">{filteredExtractions.length}</span>
                  </Pill>
                </div>
              ) : null
            }
          >
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <input className="ui-input" placeholder="search span / term / cui" value={query} onChange={(e) => setQuery(e.target.value)} />
                <Button variant="ghost" disabled={!query.trim().length} onClick={() => setQuery("")}>
                  clear
                </Button>
              </div>

              <div className="rounded-lg border border-white/10 bg-slate-950/25 p-3">
                {active ? (
                  <pre className="whitespace-pre-wrap font-mono text-xs leading-6 text-slate-100">
                    {segments.map((seg, idx) => {
                      if (!seg.extraction) return <span key={idx}>{seg.text}</span>;
                      const e = seg.extraction;
                      const key = extractionKey(e);
                      const isSelected = selectedKey === key;
                      return (
                        <mark
                          key={idx}
                          className={cx(
                            "rounded px-1 py-0.5 transition-colors",
                            groupMarkClass(e.semantic_group as SemanticGroup),
                            isSelected ? "ring-2 ring-sky-200/70" : "",
                          )}
                          onClick={() => setSelectedKey(key)}
                          title={`${e.preferred_term} (${e.cui})`}
                        >
                          {seg.text}
                          <span className="ml-1 align-top text-[10px] font-semibold uppercase tracking-wide text-white/90">
                            <span className={cx("rounded px-1 py-0.5", groupBadgeClass(e.semantic_group as SemanticGroup))}>{e.semantic_group}</span>
                            <span className={cx("ml-1 rounded px-1 py-0.5", assertionBadgeClass(e.assertion))}>{e.assertion}</span>
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
          </Card>
        </section>

        <aside className="space-y-4">
          <Card
            title="extractions"
            subtitle={active ? `${filteredExtractions.length} shown · ${active.extractions.length} total` : "no note selected"}
            right={
              results.length ? (
                <div className="flex items-center gap-2">
                  <Button variant="secondary" onClick={() => downloadCsv(results)}>
                    <DownloadSimple size={16} />
                    csv
                  </Button>
                  <Button variant="secondary" onClick={() => downloadXlsx(results)}>
                    <DownloadSimple size={16} />
                    xlsx
                  </Button>
                </div>
              ) : null
            }
          >
            <div className="space-y-3">
              <div className="grid gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  {availableAssertions.map((a) => (
                    <button
                      key={a}
                      type="button"
                      className={cx(
                        "rounded-md border px-2 py-1 text-[11px] font-semibold uppercase tracking-wide transition-opacity",
                        assertionFilter[a] ? "opacity-100" : "opacity-45",
                        assertionBadgeClass(a),
                      )}
                      onClick={() => setAssertionFilter((prev) => ({ ...prev, [a]: !prev[a] }))}
                      title="toggle assertion filter"
                    >
                      {a} <span className="font-mono tabular-nums text-white/70">{counts.byAssertion[a] ?? 0}</span>
                    </button>
                  ))}
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  {availableGroups.map((g) => (
                    <button
                      key={g}
                      type="button"
                      className={cx(
                        "rounded-md border px-2 py-1 text-[11px] font-semibold uppercase tracking-wide transition-opacity",
                        groupFilter[g] !== false ? "opacity-100" : "opacity-45",
                        groupBadgeClass(g as SemanticGroup),
                      )}
                      onClick={() => setGroupFilter((prev) => ({ ...prev, [g]: prev[g] === false }))}
                      title="toggle semantic group filter"
                    >
                      {g} <span className="font-mono tabular-nums text-white/70">{counts.byGroup[g] ?? 0}</span>
                    </button>
                  ))}
                </div>
              </div>

              <SelectionPanel selected={selected} />

              <div className="h-[420px] overflow-auto rounded-lg border border-white/10 bg-slate-950/25">
                <table className="w-full border-collapse text-left text-sm">
                  <thead className="sticky top-0 bg-slate-950/90 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                    <tr>
                      <th className="px-3 py-2">span</th>
                      <th className="px-3 py-2">term</th>
                      <th className="px-3 py-2">score</th>
                    </tr>
                  </thead>
                  <tbody className="text-slate-100">
                    {active ? (
                      filteredExtractions.map((e) => {
                        const key = extractionKey(e);
                        const isSelected = selectedKey === key;
                        return (
                          <tr
                            key={key}
                            className={cx("cursor-pointer border-t border-white/5 hover:bg-white/5", isSelected ? "bg-white/5" : "")}
                            onClick={() => setSelectedKey(key)}
                          >
                            <td className="px-3 py-2 align-top">
                              <div className="font-mono text-xs text-slate-200">{e.text}</div>
                              <div className="mt-1 flex flex-wrap items-center gap-1">
                                <span className={groupBadgeClass(e.semantic_group as SemanticGroup)}>{e.semantic_group}</span>
                                <span className={assertionBadgeClass(e.assertion)}>{e.assertion}</span>
                                <span className="rounded-md border border-white/10 bg-slate-950/40 px-2 py-1 text-[11px] font-semibold text-slate-300">{e.subject}</span>
                              </div>
                            </td>
                            <td className="px-3 py-2 align-top text-slate-200">
                              <div className="text-sm font-semibold">{e.preferred_term}</div>
                              <div className="mt-1 font-mono text-xs text-slate-400">{e.cui}</div>
                            </td>
                            <td className="px-3 py-2 align-top">
                              <div className="font-mono text-xs tabular-nums text-slate-200">{e.score.toFixed(3)}</div>
                              <div className="mt-2 h-1.5 w-24 rounded-full bg-slate-950/60">
                                <div
                                  className="h-1.5 rounded-full bg-sky-400/70"
                                  style={{ width: `${Math.max(0, Math.min(1, e.score)) * 100}%` }}
                                />
                              </div>
                            </td>
                          </tr>
                        );
                      })
                    ) : (
                      <tr>
                        <td className="px-3 py-3 text-slate-400" colSpan={3}>
                          no note selected.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </Card>
        </aside>
      </main>
    </div>
  );
}

function extractionKey(e: ConceptExtraction): string {
  return `${e.start}:${e.end}:${e.cui}:${e.assertion}:${e.subject}`;
}

function hydrateGroupFilter(prev: Record<string, boolean>, extractions: ConceptExtraction[]): Record<string, boolean> {
  const out = { ...prev };
  for (const e of extractions) if (!(e.semantic_group in out)) out[e.semantic_group] = true;
  return out;
}

function StatusPill(props: { apiOk: null | { ok: boolean; loaded: boolean } }) {
  if (!props.apiOk) return <Pill>checking api…</Pill>;
  if (!props.apiOk.ok) {
    return (
      <Pill className="border-rose-400/20 bg-rose-600/12 text-rose-100" title="api offline">
        <span className="mr-2 inline-block h-2 w-2 rounded-full bg-rose-300" />
        api offline
      </Pill>
    );
  }
  if (!props.apiOk.loaded) {
    return (
      <Pill className="border-amber-400/20 bg-amber-600/12 text-amber-100" title="api running, model not loaded yet">
        <span className="mr-2 inline-block h-2 w-2 rounded-full bg-amber-300" />
        loading model
      </Pill>
    );
  }
  return (
    <Pill className="border-emerald-400/20 bg-emerald-600/12 text-emerald-100" title="api ready">
      <span className="mr-2 inline-block h-2 w-2 rounded-full bg-emerald-300" />
      ready
    </Pill>
  );
}

function SelectionPanel(props: { selected: ConceptExtraction | null }) {
  const e = props.selected;
  if (!e) {
    return <div className="rounded-lg border border-white/10 bg-slate-950/25 px-3 py-3 text-sm text-slate-400">click a highlight or a row to inspect it.</div>;
  }
  return (
    <div className="rounded-lg border border-white/10 bg-slate-950/25 px-3 py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">selection</div>
          <div className="mt-1 truncate text-sm font-semibold text-slate-100">{e.text}</div>
          <div className="mt-0.5 truncate text-sm text-slate-300">{e.preferred_term}</div>
        </div>
        <Button
          variant="ghost"
          onClick={() => void navigator.clipboard.writeText(`${e.text}\t${e.cui}\t${e.preferred_term}`)}
        >
          <Copy size={16} />
          copy
        </Button>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className={groupBadgeClass(e.semantic_group as SemanticGroup)}>{e.semantic_group}</span>
        <span className={assertionBadgeClass(e.assertion)}>{e.assertion}</span>
        <span className="rounded-md border border-white/10 bg-slate-950/40 px-2 py-1 text-[11px] font-semibold text-slate-300">{e.subject}</span>
        <span className="rounded-md border border-white/10 bg-slate-950/40 px-2 py-1 font-mono text-[11px] tabular-nums text-slate-300">
          {e.start}–{e.end}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-md border border-white/10 bg-slate-950/40 px-2 py-2">
          <div className="ui-label">cui</div>
          <div className="mt-1 flex items-center justify-between gap-2">
            <div className="min-w-0 truncate font-mono tabular-nums text-slate-200">{e.cui}</div>
            <button
              type="button"
              className="rounded-md border border-white/10 bg-slate-950/55 px-2 py-1 text-[11px] font-semibold text-slate-200 hover:bg-slate-950/75"
              onClick={() => void navigator.clipboard.writeText(e.cui)}
              title="copy cui"
            >
              copy
            </button>
          </div>
        </div>
        <div className="rounded-md border border-white/10 bg-slate-950/40 px-2 py-2">
          <div className="ui-label">type</div>
          <div className="mt-1 font-mono tabular-nums text-slate-200">{e.tui}</div>
          <div className="mt-0.5 truncate text-slate-400">{e.semantic_type}</div>
        </div>
        <div className="rounded-md border border-white/10 bg-slate-950/40 px-2 py-2">
          <div className="ui-label">score</div>
          <div className="mt-1 font-mono tabular-nums text-slate-200">{e.score.toFixed(3)}</div>
          <div className="mt-0.5 font-mono tabular-nums text-slate-400">rerank {e.rerank_score.toFixed(3)}</div>
        </div>
        <div className="rounded-md border border-white/10 bg-slate-950/40 px-2 py-2">
          <div className="ui-label">normalized</div>
          <div className="mt-1 truncate font-mono text-slate-200">{e.normalized_text}</div>
          <div className="mt-0.5 truncate font-mono text-slate-400">{e.expanded_text}</div>
        </div>
      </div>
    </div>
  );
}

function clamp(n: number, min: number, max: number): number {
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, n));
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
