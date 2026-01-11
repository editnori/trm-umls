import mermaid from "mermaid";
import { useEffect, useMemo, useRef, useState } from "react";
import { extractSingle, getHealth } from "./api";
import { cx } from "./components/cx";
import { renderMarkdown } from "./components/markdown";
import { DEMO_EXTRACTIONS, DEMO_NOTE } from "./demo_data";
import type { Candidate, ConceptExtraction, ExtractOptions } from "./types";
import paperRaw from "../../trm_umls/paper.md?raw";

const DEFAULT_EXAMPLE_OPTIONS: ExtractOptions = {
  threshold: 0.55,
  top_k: 8,
  dedupe: true,
  clinical_rerank: true,
  rerank: true,
  include_candidates: true,
  relation_rerank: false,
  lexical_weight: 0.3,
  rerank_margin: 0.04,
};

type ApiState =
  | { kind: "checking" }
  | { kind: "offline" }
  | { kind: "loading" }
  | { kind: "ready" };

type Tab = "input" | "mentions" | "candidates" | "output";

function splitPaper(md: string): [string, string] {
  const marker = "\n---\n";
  const idx = md.indexOf(marker);
  if (idx === -1) return [md, ""];
  return [md.slice(0, idx), md.slice(idx)];
}

export default function Methodology() {
  const [api, setApi] = useState<ApiState>({ kind: "checking" });
  const [live, setLive] = useState(true);

  const [tab, setTab] = useState<Tab>("input");
  const [exampleText, setExampleText] = useState(DEMO_NOTE);
  const [options, setOptions] = useState<ExtractOptions>(DEFAULT_EXAMPLE_OPTIONS);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<ConceptExtraction[]>(DEMO_EXTRACTIONS);
  const [selected, setSelected] = useState<string | null>(null);

  const articleRef = useRef<HTMLDivElement | null>(null);

  const [introMd, restMd] = useMemo(() => splitPaper(paperRaw), []);
  const introHtml = useMemo(() => patchAnchors(renderMarkdown(introMd)), [introMd]);
  const restHtml = useMemo(() => patchAnchors(renderMarkdown(restMd)), [restMd]);

  useEffect(() => {
    let mounted = true;
    async function tick() {
      try {
        const h = await getHealth();
        if (!mounted) return;
        if (!h.ok) setApi({ kind: "offline" });
        else if (!h.loaded) setApi({ kind: "loading" });
        else setApi({ kind: "ready" });
      } catch {
        if (mounted) setApi({ kind: "offline" });
      }
    }
    void tick();
    const id = window.setInterval(tick, 5000);
    return () => {
      mounted = false;
      window.clearInterval(id);
    };
  }, []);

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "neutral",
      securityLevel: "strict",
    });
  }, []);

  useEffect(() => {
    if (!articleRef.current) return;
    const nodes = Array.from(articleRef.current.querySelectorAll(".mermaid")) as HTMLElement[];
    if (nodes.length === 0) return;
    mermaid.run({ nodes });
  }, [introHtml, restHtml]);

  useEffect(() => {
    const jump = () => {
      const hash = window.location.hash.slice(1);
      const parts = hash.split("/");
      if (parts[0] !== "methodology") return;
      const anchor = parts[1];
      if (!anchor) return;
      const el = document.getElementById(anchor);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    };
    jump();
    window.addEventListener("hashchange", jump);
    return () => window.removeEventListener("hashchange", jump);
  }, []);

  const apiLabel = useMemo(() => {
    if (api.kind === "checking") return "checking api";
    if (api.kind === "offline") return "demo (api offline)";
    if (api.kind === "loading") return "api loading model";
    return "live";
  }, [api.kind]);

  const canRunLive = api.kind === "ready";
  const usingLive = live && canRunLive;

  const selectedRow = useMemo(() => {
    if (!selected) return null;
    return rows.find((r) => rowKey(r) === selected) ?? null;
  }, [rows, selected]);

  const candidates = useMemo(() => {
    const c = selectedRow?.candidates ?? null;
    if (!c || !c.length) return null;
    return c as Candidate[];
  }, [selectedRow]);

  async function runExample(opts?: { auto?: boolean }) {
    setBusy(true);
    if (!opts?.auto) setError(null);
    try {
      if (!usingLive) {
        setRows(DEMO_EXTRACTIONS);
        if (!selected) setSelected(rowKey(DEMO_EXTRACTIONS[0]));
        return;
      }
      const res = await extractSingle(exampleText, options);
      setRows(res.extractions);
      if (!selected && res.extractions.length) setSelected(rowKey(res.extractions[0]));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
      setRows(DEMO_EXTRACTIONS);
      if (!selected) setSelected(rowKey(DEMO_EXTRACTIONS[0]));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen bg-page">
      <header className="flex h-9 items-center px-4">
        <div className="flex items-center gap-2 text-[11px] text-muted">
          <span className="text-body font-medium">trm-umls</span>
          <span className="text-faint">·</span>
          <span>{apiLabel}</span>
        </div>
        <div className="ml-auto flex items-center gap-3 text-[11px] text-muted">
          <a href="/" className="hover:text-primary">extractor</a>
          <span className="text-faint">·</span>
          <a href="#methodology/abstract" className="hover:text-primary">paper</a>
        </div>
      </header>

      <article ref={articleRef} className="prose mx-auto max-w-3xl px-6 py-10">
        <div dangerouslySetInnerHTML={{ __html: introHtml }} />

        <section className="my-10 rounded border border-border bg-surface">
          <div className="flex items-center gap-2 border-b border-border px-4 py-2 text-[10px] uppercase tracking-wide text-muted">
            try it now
            <span className="text-faint">·</span>
            <span className="normal-case text-[11px]">{apiLabel}</span>
            <div className="ml-auto flex items-center gap-2 text-[10px]">
              <button
                className={cx(
                  "rounded px-2 py-1 text-[10px]",
                  usingLive ? "bg-elevated text-primary" : "text-muted hover:text-primary",
                )}
                onClick={() => setLive(true)}
              >
                live
              </button>
              <button
                className={cx(
                  "rounded px-2 py-1 text-[10px]",
                  !live ? "bg-elevated text-primary" : "text-muted hover:text-primary",
                )}
                onClick={() => setLive(false)}
              >
                demo
              </button>
              <span className="text-faint">·</span>
              <button
                className="rounded px-2 py-1 text-[10px] text-muted hover:text-primary"
                onClick={() => void runExample()}
                disabled={busy}
              >
                {busy ? "running…" : "run"}
              </button>
            </div>
          </div>

          <div className="border-b border-border bg-elevated px-4 py-2">
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              {(
                [
                  ["input", "1. input"],
                  ["mentions", "2. mentions"],
                  ["candidates", "3. retrieval"],
                  ["output", "4. output"],
                ] as Array<[Tab, string]>
              ).map(([k, label]) => (
                <button
                  key={k}
                  onClick={() => setTab(k)}
                  className={cx(
                    "rounded px-2.5 py-1 text-[11px] font-medium",
                    tab === k ? "bg-surface text-primary" : "text-muted hover:text-primary",
                  )}
                >
                  {label}
                </button>
              ))}
              <div className="ml-auto text-[10px] text-faint">
                threshold <span className="font-mono">{options.threshold.toFixed(2)}</span> · top-k{" "}
                <span className="font-mono">{options.top_k}</span>
              </div>
            </div>
          </div>

          <div className="bg-elevated px-4 py-4">
            {tab === "input" ? (
              <div>
                <div className="section-label">note text</div>
                <textarea
                  value={exampleText}
                  onChange={(e) => setExampleText(e.target.value)}
                  rows={4}
                  className="mt-2 w-full resize-none rounded border border-border bg-elevated px-3 py-2 font-mono text-[12px] leading-relaxed text-primary outline-none focus:border-border-strong"
                  spellCheck={false}
                />
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div>
                    <div className="section-label">threshold</div>
                    <input
                      type="range"
                      min={0.35}
                      max={0.9}
                      step={0.01}
                      value={options.threshold}
                      onChange={(e) => setOptions((o) => ({ ...o, threshold: Number(e.target.value) }))}
                      className="mt-2 w-full"
                    />
                  </div>
                  <div>
                    <div className="section-label">lexical weight</div>
                    <input
                      type="range"
                      min={0}
                      max={0.8}
                      step={0.01}
                      value={options.lexical_weight ?? 0.3}
                      onChange={(e) => setOptions((o) => ({ ...o, lexical_weight: Number(e.target.value) }))}
                      className="mt-2 w-full"
                    />
                  </div>
                </div>
                <div className="mt-3 text-[11px] text-muted">
                  if the api is offline, this panel uses a small demo output.
                </div>
              </div>
            ) : null}

            {tab === "mentions" ? (
              <div>
                <div className="section-label">extracted spans</div>
                <div className="mt-2 space-y-2">
                  {rows.length ? (
                    rows.map((r) => {
                      const key = rowKey(r);
                      const on = selected === key;
                      return (
                        <button
                          key={key}
                          className={cx(
                            "w-full rounded border px-3 py-2 text-left transition-colors",
                            on ? "border-border-strong bg-surface" : "border-border bg-elevated hover:bg-hover",
                          )}
                          onClick={() => setSelected(key)}
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="min-w-0">
                              <div className="truncate font-mono text-[12px] text-primary">{r.text}</div>
                              <div className="mt-0.5 truncate text-[11px] text-muted">
                                normalized: <span className="font-mono">{r.normalized_text}</span>
                              </div>
                            </div>
                            <div className="shrink-0 text-right">
                              <div className="font-mono text-[11px] text-muted">{r.score.toFixed(3)}</div>
                              <div className="mt-0.5 text-[10px] text-faint">
                                {r.semantic_group} · {r.assertion}
                              </div>
                            </div>
                          </div>
                        </button>
                      );
                    })
                  ) : (
                    <div className="text-[11px] text-muted">no spans returned.</div>
                  )}
                </div>
              </div>
            ) : null}

            {tab === "candidates" ? (
              <div>
                <div className="section-label">retrieval + rerank</div>
                {!selectedRow ? (
                  <div className="mt-2 text-[11px] text-muted">select a span in “mentions”.</div>
                ) : !candidates ? (
                  <div className="mt-2 text-[11px] text-muted">
                    no candidate list available. enable candidates and rerun in the extractor.
                  </div>
                ) : (
                  <>
                    <div className="mt-2">
                      <div className="font-mono text-[12px] text-primary">{selectedRow.text}</div>
                      <div className="text-[11px] text-muted">
                        faiss top-k results, then rerank inside a small margin.
                      </div>
                    </div>
                    <div className="mt-3 overflow-hidden rounded border border-border">
                      <table className="w-full text-[11px]">
                        <thead className="bg-surface text-[10px] uppercase tracking-wide text-muted">
                          <tr>
                            <th className="px-3 py-2 text-left">rank</th>
                            <th className="px-3 py-2 text-left">cui</th>
                            <th className="px-3 py-2 text-left">preferred term</th>
                            <th className="px-3 py-2 text-right">sim</th>
                            <th className="px-3 py-2 text-right">lex</th>
                            <th className="px-3 py-2 text-right">bias</th>
                            <th className="px-3 py-2 text-right">pen</th>
                            <th className="px-3 py-2 text-right">rerank</th>
                          </tr>
                        </thead>
                        <tbody className="font-mono">
                          {candidates.map((c) => (
                            <tr key={`${c.rank}:${c.cui}`} className="border-t border-border">
                              <td className="px-3 py-2 text-muted">{c.rank}</td>
                              <td className="px-3 py-2 text-muted">{c.cui}</td>
                              <td className="px-3 py-2 font-sans text-primary">
                                {c.preferred_term}
                                <span className="ml-2 text-[10px] text-faint">
                                  {c.tui} · {c.semantic_group}
                                </span>
                              </td>
                              <td className="px-3 py-2 text-right text-muted">{c.score.toFixed(3)}</td>
                              <td className="px-3 py-2 text-right text-muted">{c.lex.toFixed(3)}</td>
                              <td className="px-3 py-2 text-right text-muted">{c.bias.toFixed(3)}</td>
                              <td className="px-3 py-2 text-right text-muted">{c.penalty.toFixed(3)}</td>
                              <td className="px-3 py-2 text-right text-primary">{c.rerank_score.toFixed(3)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>
            ) : null}

            {tab === "output" ? (
              <div>
                <div className="section-label">structured output</div>
                <div className="mt-2 overflow-hidden rounded border border-border">
                  <table className="w-full text-[11px]">
                    <thead className="bg-surface text-[10px] uppercase tracking-wide text-muted">
                      <tr>
                        <th className="px-3 py-2 text-left">span</th>
                        <th className="px-3 py-2 text-left">cui</th>
                        <th className="px-3 py-2 text-left">assertion</th>
                        <th className="px-3 py-2 text-right">score</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono">
                      {rows.map((r) => (
                        <tr key={rowKey(r)} className="border-t border-border">
                          <td className="px-3 py-2 text-primary">{r.text}</td>
                          <td className="px-3 py-2 text-muted">{r.cui}</td>
                          <td className="px-3 py-2 text-muted">{r.assertion}</td>
                          <td className="px-3 py-2 text-right text-muted">{r.score.toFixed(3)}</td>
                        </tr>
                      ))}
                      {!rows.length ? (
                        <tr>
                          <td className="px-3 py-3 text-[11px] text-muted" colSpan={4}>
                            no extractions returned.
                          </td>
                        </tr>
                      ) : null}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
          </div>

          {api.kind === "offline" ? (
            <div className="border-t border-border bg-surface px-4 py-3 text-[11px] text-muted">
              api not reachable. start it with{" "}
              <span className="font-mono text-primary">python3 -m trm_umls.api</span>. this page is showing demo output.
            </div>
          ) : null}

          {error ? (
            <div className="border-t border-border bg-surface px-4 py-3 text-[11px] text-[var(--error)]">
              {error}
            </div>
          ) : null}
        </section>

        <div dangerouslySetInnerHTML={{ __html: restHtml }} />
      </article>
    </div>
  );
}

function rowKey(r: ConceptExtraction): string {
  return `${r.start}:${r.end}:${r.cui}:${r.assertion}:${r.subject}`;
}

function patchAnchors(html: string): string {
  return html.replace(/href="#([^"]+)"/g, 'href="#methodology/$1"');
}
