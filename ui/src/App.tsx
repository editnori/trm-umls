import { ArrowsLeftRight, CloudArrowUp, Copy, DownloadSimple, Moon, Play, Pulse, Sun, WarningCircle } from "@phosphor-icons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { extractBatch, extractSingle, getHealth } from "./api";
import { cx } from "./components/cx";
import { Button, Stepper, Switch, Textarea } from "./components/ui";
import { downloadCsv, downloadXlsx } from "./export";
import type { Assertion, ConceptExtraction, ExtractOptions, NoteResult } from "./types";
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
  const [darkMode, setDarkMode] = useState(true);
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
  
  // Toggle to show preferred terms instead of original spans
  const [showPreferred, setShowPreferred] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", darkMode ? "dark" : "light");
  }, [darkMode]);

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

  const apiStatus = !apiOk ? "checking" : !apiOk.ok ? "offline" : !apiOk.loaded ? "loading" : "ready";
  const statusDot = {
    checking: "bg-yellow-500",
    offline: "bg-red-500",
    loading: "bg-yellow-500",
    ready: "bg-emerald-500",
  }[apiStatus];

  return (
    <div className="h-full bg-page flex flex-col">
      {/* Header - blends with page */}
      <header className="flex items-center h-9 px-4 shrink-0">
        {/* Left: name + status */}
        <div className="flex items-center gap-2 text-[11px]">
          <span className="text-muted font-medium">trm-umls</span>
          <span className={cx("w-1.5 h-1.5 rounded-full", statusDot)} />
        </div>
        
        {/* Right: methodology link + theme toggle */}
        <div className="ml-auto flex items-center gap-3">
          <a
            href="#methodology"
            className="text-[10px] text-faint hover:text-muted transition-colors"
          >
            methodology
          </a>
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="p-1 text-faint hover:text-muted transition-colors"
            title={darkMode ? "Light mode" : "Dark mode"}
          >
            {darkMode ? <Sun size={13} /> : <Moon size={13} />}
          </button>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 grid lg:grid-cols-[280px_1fr_320px] divide-x divide-border overflow-hidden min-h-0">
        
        {/* Left - Controls */}
        <aside className="min-w-0 min-h-0 flex flex-col overflow-hidden bg-surface">
          <div className="flex-1 overflow-auto p-4 space-y-4">
            <div>
              <label className="block text-[10px] font-semibold text-muted uppercase tracking-wider mb-2">Input</label>
              <Textarea
                ref={textareaRef}
                value={text}
                onChange={setText}
                rows={8}
                placeholder="paste clinical text…"
                onKeyDown={(e) => {
                  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") void runFromText();
                }}
              />
            </div>

            <div className="space-y-1">
              <label className="block text-[10px] font-semibold text-muted uppercase tracking-wider mb-2">Parameters</label>
              <Stepper
                label="threshold"
                value={options.threshold}
                onChange={(v) => setOptions((o) => ({ ...o, threshold: round2(v) }))}
                step={0.05}
                min={0}
                max={1}
                format={(v) => v.toFixed(2)}
              />
              <Stepper
                label="top-k"
                value={options.top_k}
                onChange={(v) => setOptions((o) => ({ ...o, top_k: Math.round(v) }))}
                step={5}
                min={1}
                max={50}
              />
            </div>

            <div className="space-y-1">
              <label className="block text-[10px] font-semibold text-muted uppercase tracking-wider mb-2">Options</label>
              <Switch
                label="clinical rerank"
                checked={options.clinical_rerank}
                onChange={(v) => setOptions((o) => ({ ...o, clinical_rerank: v }))}
              />
              <Switch
                label="dedupe"
                checked={options.dedupe}
                onChange={(v) => setOptions((o) => ({ ...o, dedupe: v }))}
              />
              <Switch
                label="candidates"
                checked={options.include_candidates}
                onChange={(v) => setOptions((o) => ({ ...o, include_candidates: v }))}
              />
            </div>

            <div className="flex items-center gap-2 pt-3 border-t border-border">
              <Button
                variant="primary"
                disabled={busy || text.trim().length < 1 || apiOk?.ok === false}
                onClick={runFromText}
              >
                {busy ? <Pulse size={14} className="animate-pulse" /> : <Play size={14} weight="fill" />}
                Run
              </Button>
              <Button
                variant="secondary"
                disabled={busy || apiOk?.ok === false}
                onClick={() => fileInputRef.current?.click()}
              >
                <CloudArrowUp size={14} />
                Upload
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
            </div>

            {error && (
              <div className="flex items-start gap-2 p-3 rounded bg-red-500/10 text-red-500 text-xs">
                <WarningCircle size={14} weight="fill" className="mt-0.5 shrink-0" />
                <span className="font-mono">{error}</span>
              </div>
            )}

            {results.length > 0 && (
              <div className="pt-3 border-t border-border">
                <label className="block text-[10px] font-semibold text-muted uppercase tracking-wider mb-2">Documents</label>
                <div className="space-y-1">
                  {results.map((r) => (
                    <button
                      key={r.id}
                      className={cx(
                        "w-full text-left px-3 py-2 rounded text-xs transition-colors",
                        r.id === activeId ? "bg-accent/15 text-primary" : "text-body hover:bg-hover"
                      )}
                      onClick={() => {
                        setActiveId(r.id);
                        setSelectedKey(null);
                        setGroupFilter((prev) => hydrateGroupFilter(prev, r.extractions));
                      }}
                    >
                      <div className="flex justify-between items-center">
                        <span className="truncate font-medium">{r.name}</span>
                        <span className="font-mono text-muted">{r.extractions.length}</span>
                      </div>
                    </button>
                  ))}
                </div>
                <button
                  className="mt-2 text-xs text-muted hover:text-primary"
                  onClick={() => {
                    setResults([]);
                    setActiveId(null);
                    setSelectedKey(null);
                    setGroupFilter({});
                  }}
                >
                  Clear all
                </button>
              </div>
            )}
          </div>
        </aside>

        {/* Center - Document */}
        <section className="min-w-0 min-h-0 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-auto p-4">
            {active ? (
              <>
                {/* Header with view toggle */}
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h2 className="text-sm font-semibold text-primary">{active.name}</h2>
                    <span className="text-xs text-muted font-mono">
                      {filteredExtractions.length} extractions · {String(active.meta?.ms ?? "—")}ms
                    </span>
                  </div>
                  
                  {/* View toggle: Original ↔ Preferred */}
                  <button
                    onClick={() => setShowPreferred(!showPreferred)}
                    className={cx(
                      "flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-medium transition-colors",
                      showPreferred 
                        ? "bg-[var(--text-primary)] text-[var(--bg-page)]" 
                        : "bg-[var(--bg-surface)] border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                    )}
                    title={showPreferred ? "Showing preferred terms" : "Showing original text"}
                  >
                    <ArrowsLeftRight size={12} />
                    {showPreferred ? "Preferred" : "Original"}
                  </button>
                </div>
                
                {/* Filter */}
                <input
                  className="w-full px-3 py-2 mb-4 text-xs bg-elevated border border-border rounded 
                             focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent
                             text-primary placeholder:text-muted"
                  placeholder="Filter by span, term, or CUI…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />

                {/* Document view */}
                <div className="bg-elevated rounded border border-border p-4">
                  <div className="font-mono text-sm leading-7 text-primary whitespace-pre-wrap">
                    {segments.map((seg, idx) => {
                      if (!seg.extraction) return <span key={idx}>{seg.text}</span>;
                      const e = seg.extraction;
                      const key = extractionKey(e);
                      const isSelected = selectedKey === key;
                      const displayText = showPreferred ? e.preferred_term : seg.text;
                      
                      return (
                        <span
                          key={idx}
                          className={cx(
                            "entity-mark",
                            `entity-${e.semantic_group}`,
                            isSelected && "selected"
                          )}
                          onClick={() => setSelectedKey(key)}
                          title={showPreferred ? `Original: ${seg.text}` : `Preferred: ${e.preferred_term}`}
                        >
                          {displayText}
                          <span className={cx("entity-badge", `badge-${e.semantic_group}`)}>
                            {e.semantic_group}
                          </span>
                        </span>
                      );
                    })}
                  </div>
                </div>
                
                {/* Legend when showing preferred */}
                {showPreferred && (
                  <p className="mt-2 text-xs text-muted italic">
                    ↑ Showing standardized terminology. Hover spans to see original text.
                  </p>
                )}
              </>
            ) : (
              <div className="h-full flex items-center justify-center">
                <p className="text-sm text-muted">Paste clinical text and click Run</p>
              </div>
            )}
          </div>
        </section>

        {/* Right - Results */}
        <aside className="min-w-0 min-h-0 flex flex-col overflow-hidden bg-surface">
          <div className="flex-1 overflow-auto p-4">
            {/* Filters */}
            {(availableAssertions.length > 0 || availableGroups.length > 0) && (
              <div className="mb-4 pb-4 border-b border-border">
                <label className="block text-[10px] font-semibold text-muted uppercase tracking-wider mb-2">Filters</label>
                
                {/* Assertion filters */}
                <div className="flex flex-wrap gap-1.5 mb-2">
                  {availableAssertions.map((a) => (
                    <button
                      key={a}
                      className={cx(
                        "filter-pill",
                        `filter-${a}`,
                        !assertionFilter[a] && "inactive"
                      )}
                      onClick={() => setAssertionFilter((prev) => ({ ...prev, [a]: !prev[a] }))}
                    >
                      {a}
                      <span className="count">{counts.byAssertion[a] ?? 0}</span>
                    </button>
                  ))}
                </div>
                
                {/* Group filters */}
                <div className="flex flex-wrap gap-1.5">
                  {availableGroups.map((g) => (
                    <button
                      key={g}
                      className={cx(
                        "filter-pill",
                        `filter-${g}`,
                        groupFilter[g] === false && "inactive"
                      )}
                      onClick={() => setGroupFilter((prev) => ({ ...prev, [g]: prev[g] === false }))}
                    >
                      {g}
                      <span className="count">{counts.byGroup[g] ?? 0}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Selected */}
            {selected && (
              <div className="mb-4 p-3 rounded bg-elevated border border-border">
                <div className="flex justify-between items-start mb-2">
                  <div className="min-w-0 flex-1">
                    <div className="font-semibold text-primary text-sm">{selected.text}</div>
                    <div className="text-xs text-accent mt-0.5">→ {selected.preferred_term}</div>
                  </div>
                  <button
                    className="text-muted hover:text-primary transition-colors ml-2"
                    onClick={() => void navigator.clipboard.writeText(`${selected.text}\t${selected.cui}`)}
                    title="Copy"
                  >
                    <Copy size={14} />
                  </button>
                </div>
                <div className="flex flex-wrap gap-2 text-[10px] text-muted font-mono mt-2">
                  <span className="px-2 py-1 bg-surface rounded">{selected.cui}</span>
                  <span className="px-2 py-1 bg-surface rounded">{selected.tui}</span>
                  <span className="px-2 py-1 bg-surface rounded">{selected.score.toFixed(3)}</span>
                </div>
              </div>
            )}

            {/* Export */}
            {results.length > 0 && (
              <div className="flex gap-2 mb-4">
                <Button variant="secondary" onClick={() => downloadCsv(results)}>
                  <DownloadSimple size={12} /> CSV
                </Button>
                <Button variant="secondary" onClick={() => downloadXlsx(results)}>
                  <DownloadSimple size={12} /> XLSX
                </Button>
              </div>
            )}

            {/* Table */}
            {filteredExtractions.length > 0 && (
              <div>
                <label className="block text-[10px] font-semibold text-muted uppercase tracking-wider mb-2">
                  Results ({filteredExtractions.length})
                </label>
                <div className="rounded border border-border overflow-hidden">
                  <table className="w-full text-xs">
                    <thead className="bg-elevated border-b border-border">
                      <tr>
                        <th className="text-left px-3 py-2 text-[10px] font-semibold text-muted uppercase">Span</th>
                        <th className="text-left px-3 py-2 text-[10px] font-semibold text-muted uppercase">Preferred</th>
                        <th className="text-right px-3 py-2 text-[10px] font-semibold text-muted uppercase">Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredExtractions.map((e) => {
                        const key = extractionKey(e);
                        const isSelected = selectedKey === key;
                        return (
                          <tr
                            key={key}
                            className={cx(
                              "border-b border-border/50 cursor-pointer transition-colors",
                              isSelected ? "bg-accent/10" : "hover:bg-hover"
                            )}
                            onClick={() => setSelectedKey(key)}
                          >
                            <td className="px-3 py-2">
                              <div className="font-medium text-primary text-xs">{e.text}</div>
                              <div className="flex gap-1 mt-1">
                                <span className={cx("entity-badge", `badge-${e.semantic_group}`)}>{e.semantic_group}</span>
                                <span className={cx("entity-badge", `badge-${e.assertion}`)}>{e.assertion}</span>
                              </div>
                            </td>
                            <td className="px-3 py-2 text-muted text-xs truncate max-w-[100px]">
                              {e.preferred_term}
                            </td>
                            <td className="px-3 py-2 text-right font-mono text-muted tabular-nums">
                              {e.score.toFixed(2)}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
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

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
