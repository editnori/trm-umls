import { Play, Pulse, WarningCircle } from "@phosphor-icons/react";
import { useEffect, useMemo, useState } from "react";
import { extractSingle, getHealth } from "./api";
import { cx } from "./components/cx";
import { Latex } from "./components/latex";
import { DEMO_EXTRACTIONS, DEMO_NOTE } from "./demo_data";
import type { Candidate, ConceptExtraction, ExtractOptions } from "./types";

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

type Tab = "input" | "mentions" | "candidates" | "output" | "math";

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

  const apiLabel = useMemo(() => {
    if (api.kind === "checking") return "checking api";
    if (api.kind === "offline") return "demo (api offline)";
    if (api.kind === "loading") return "api loading model";
    return "live";
  }, [api.kind]);

  const canRunLive = api.kind === "ready";
  const usingLive = live && canRunLive;

  useEffect(() => {
    // best-effort auto-run once when the api becomes ready
    if (api.kind !== "ready") return;
    if (!live) return;
    void runExample({ auto: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api.kind]);

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
    <div className="min-h-screen bg-[var(--bg-page)]">
      <header className="border-b border-[var(--border)] px-6 py-3">
        <div className="mx-auto flex max-w-3xl items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-[var(--text-muted)]">trm-umls</span>
            <span className="text-[10px] text-[var(--text-faint)]">·</span>
            <span className="text-[11px] text-[var(--text-faint)]">{apiLabel}</span>
          </div>
          <a
            href="/"
            className="text-[11px] text-[var(--text-muted)] hover:text-[var(--text-primary)]"
          >
            ← extractor
          </a>
        </div>
      </header>

      <article className="prose mx-auto max-w-3xl px-6 py-12">
        <h1 className="mb-2 text-2xl font-semibold text-[var(--text-primary)]">
          How trm-umls works
        </h1>
        <p className="mb-8 text-sm text-[var(--text-muted)]">
          A small embedding-distilled UMLS concept linker for clinical text, built to stay local.
        </p>

        <hr />

        <h2>what you are trying to do</h2>
        <p>
          You start with free text. You want to end with a{" "}
          <strong>span</strong> (where in the note the text came from) and a{" "}
          <strong>concept id</strong> (a UMLS CUI), plus a few clinical attributes like
          whether it is present vs denied.
        </p>
        <p>
          The hard part is that clinical notes are messy. They contain abbreviations, partial
          phrases, and context that flips meaning:
        </p>
        <pre>
          <code>{`"Denies chest pain"  → chest pain (ABSENT)
"FHx: breast cancer" → breast cancer (FAMILY)
"COPD" → chronic obstructive pulmonary disease`}</code>
        </pre>

        <h2>the core idea (retrieval-first)</h2>
        <p>
          trm-umls does not try to cram UMLS into a huge generative model. Instead it uses
          a <em>tiny retriever</em> that maps text into a high-quality embedding space, then
          uses nearest-neighbor search to look up the closest UMLS concept.
        </p>
        <p>
          The teacher embedding space is <code>SapBERT</code>. We embed every UMLS concept once,
          build a FAISS index, then train a small student encoder (about 9.8M parameters) to
          land in the same neighborhood as the teacher for short clinical spans.
        </p>

        <h2>try it now (live when available)</h2>
        <p>
          This panel runs locally. When the api is connected it uses the real pipeline. When it
          is not, it falls back to a small offline demo so you can still see the flow.
        </p>

        <div className="my-6 overflow-hidden rounded border border-[var(--border)]">
          <div className="flex items-center gap-3 border-b border-[var(--border)] bg-[var(--bg-surface)] px-4 py-2">
            <div className="text-[11px] font-semibold text-[var(--text-primary)]">
              walkthrough
            </div>
            <div className="ml-auto flex items-center gap-2">
              <button
                className={cx(
                  "rounded px-2 py-1 text-[10px] font-medium transition-colors",
                  usingLive ? "bg-[var(--text-primary)] text-[var(--bg-page)]" : "text-[var(--text-muted)] hover:text-[var(--text-primary)]",
                )}
                onClick={() => setLive(true)}
                title="use the local api when available"
              >
                live
              </button>
              <button
                className={cx(
                  "rounded px-2 py-1 text-[10px] font-medium transition-colors",
                  !live ? "bg-[var(--text-primary)] text-[var(--bg-page)]" : "text-[var(--text-muted)] hover:text-[var(--text-primary)]",
                )}
                onClick={() => setLive(false)}
                title="force demo mode"
              >
                demo
              </button>
              <span className="text-[10px] text-[var(--text-faint)]">·</span>
              <button
                className="inline-flex items-center gap-1 rounded px-2 py-1 text-[10px] font-medium text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                onClick={() => void runExample()}
                disabled={busy}
              >
                {busy ? <Pulse size={12} className="animate-pulse" /> : <Play size={12} weight="fill" />}
                run
              </button>
            </div>
          </div>

          <div className="border-b border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-3">
            <div className="flex flex-wrap items-center gap-2">
              {(
                [
                  ["input", "1. input"],
                  ["mentions", "2. mentions"],
                  ["candidates", "3. retrieval"],
                  ["output", "4. output"],
                  ["math", "5. math"],
                ] as Array<[Tab, string]>
              ).map(([k, label]) => (
                <button
                  key={k}
                  onClick={() => setTab(k)}
                  className={cx(
                    "rounded px-2.5 py-1 text-[11px] font-medium transition-colors",
                    tab === k
                      ? "bg-[var(--bg-surface)] text-[var(--text-primary)]"
                      : "text-[var(--text-muted)] hover:text-[var(--text-primary)]",
                  )}
                >
                  {label}
                </button>
              ))}
              <div className="ml-auto text-[10px] text-[var(--text-faint)]">
                threshold <span className="font-mono">{options.threshold.toFixed(2)}</span> · top-k{" "}
                <span className="font-mono">{options.top_k}</span>
              </div>
            </div>
          </div>

          <div className="bg-[var(--bg-elevated)] p-4">
            {tab === "input" ? (
              <div>
                <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                  note text
                </div>
                <textarea
                  value={exampleText}
                  onChange={(e) => setExampleText(e.target.value)}
                  rows={4}
                  className="w-full resize-none rounded border border-[var(--border)] bg-[var(--bg-elevated)] px-3 py-2 font-mono text-[12px] leading-relaxed text-[var(--text-primary)] outline-none focus:border-[var(--border-strong)]"
                  spellCheck={false}
                />
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div>
                    <div className="text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                      threshold
                    </div>
                    <input
                      type="range"
                      min={0.35}
                      max={0.9}
                      step={0.01}
                      value={options.threshold}
                      onChange={(e) =>
                        setOptions((o) => ({ ...o, threshold: Number(e.target.value) }))
                      }
                      className="mt-2 w-full"
                    />
                  </div>
                  <div>
                    <div className="text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                      lexical weight
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={0.8}
                      step={0.01}
                      value={options.lexical_weight ?? 0.3}
                      onChange={(e) =>
                        setOptions((o) => ({ ...o, lexical_weight: Number(e.target.value) }))
                      }
                      className="mt-2 w-full"
                    />
                  </div>
                </div>
                <div className="mt-3 text-[11px] text-[var(--text-muted)]">
                  tip: turn on <span className="font-mono">candidates</span> in the extractor UI when you want to inspect reranking.
                </div>
              </div>
            ) : null}

            {tab === "mentions" ? (
              <div>
                <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                  extracted spans
                </div>
                <div className="space-y-2">
                  {rows.length ? (
                    rows.map((r) => {
                      const key = rowKey(r);
                      const on = selected === key;
                      return (
                        <button
                          key={key}
                          className={cx(
                            "w-full rounded border px-3 py-2 text-left transition-colors",
                            on
                              ? "border-[var(--border-strong)] bg-[var(--bg-surface)]"
                              : "border-[var(--border)] bg-[var(--bg-elevated)] hover:bg-[var(--bg-hover)]",
                          )}
                          onClick={() => setSelected(key)}
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="min-w-0">
                              <div className="truncate font-mono text-[12px] text-[var(--text-primary)]">
                                {r.text}
                              </div>
                              <div className="mt-0.5 truncate text-[11px] text-[var(--text-muted)]">
                                normalized: <span className="font-mono">{r.normalized_text}</span>
                              </div>
                            </div>
                            <div className="shrink-0 text-right">
                              <div className="font-mono text-[11px] text-[var(--text-muted)]">
                                {r.score.toFixed(3)}
                              </div>
                              <div className="mt-0.5 text-[10px] text-[var(--text-faint)]">
                                {r.semantic_group} · {r.assertion}
                              </div>
                            </div>
                          </div>
                        </button>
                      );
                    })
                  ) : (
                    <div className="text-[11px] text-[var(--text-muted)]">no spans returned.</div>
                  )}
                </div>
              </div>
            ) : null}

            {tab === "candidates" ? (
              <div>
                <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                  retrieval + rerank
                </div>
                {!selectedRow ? (
                  <div className="text-[11px] text-[var(--text-muted)]">select a span in “mentions”.</div>
                ) : !candidates ? (
                  <div className="text-[11px] text-[var(--text-muted)]">
                    no candidate list available for this span. enable candidates and rerun.
                  </div>
                ) : (
                  <>
                    <div className="mb-3">
                      <div className="font-mono text-[12px] text-[var(--text-primary)]">{selectedRow.text}</div>
                      <div className="text-[11px] text-[var(--text-muted)]">
                        we embed the normalized span, run FAISS top-k, then rerank inside a small margin.
                      </div>
                    </div>

                    <div className="overflow-hidden rounded border border-[var(--border)]">
                      <table className="w-full text-[11px]">
                        <thead className="bg-[var(--bg-surface)] text-[10px] uppercase tracking-wide text-[var(--text-muted)]">
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
                            <tr key={`${c.rank}:${c.cui}`} className="border-t border-[var(--border)]">
                              <td className="px-3 py-2 text-[var(--text-muted)]">{c.rank}</td>
                              <td className="px-3 py-2 text-[var(--text-muted)]">{c.cui}</td>
                              <td className="px-3 py-2 font-sans text-[var(--text-primary)]">
                                {c.preferred_term}
                                <span className="ml-2 text-[10px] text-[var(--text-faint)]">
                                  {c.tui} · {c.semantic_group}
                                </span>
                              </td>
                              <td className="px-3 py-2 text-right text-[var(--text-muted)]">{c.score.toFixed(3)}</td>
                              <td className="px-3 py-2 text-right text-[var(--text-muted)]">{c.lex.toFixed(3)}</td>
                              <td className="px-3 py-2 text-right text-[var(--text-muted)]">{c.bias.toFixed(3)}</td>
                              <td className="px-3 py-2 text-right text-[var(--text-muted)]">{c.penalty.toFixed(3)}</td>
                              <td className="px-3 py-2 text-right text-[var(--text-primary)]">{c.rerank_score.toFixed(3)}</td>
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
                <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                  structured output
                </div>
                <div className="overflow-hidden rounded border border-[var(--border)]">
                  <table className="w-full text-[11px]">
                    <thead className="bg-[var(--bg-surface)] text-[10px] uppercase tracking-wide text-[var(--text-muted)]">
                      <tr>
                        <th className="px-3 py-2 text-left">span</th>
                        <th className="px-3 py-2 text-left">cui</th>
                        <th className="px-3 py-2 text-left">assertion</th>
                        <th className="px-3 py-2 text-right">score</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono">
                      {rows.map((r) => (
                        <tr key={rowKey(r)} className="border-t border-[var(--border)]">
                          <td className="px-3 py-2 text-[var(--text-primary)]">{r.text}</td>
                          <td className="px-3 py-2 text-[var(--text-muted)]">{r.cui}</td>
                          <td className="px-3 py-2 text-[var(--text-muted)]">{r.assertion}</td>
                          <td className="px-3 py-2 text-right text-[var(--text-muted)]">{r.score.toFixed(3)}</td>
                        </tr>
                      ))}
                      {!rows.length ? (
                        <tr>
                          <td className="px-3 py-3 text-[11px] text-[var(--text-muted)]" colSpan={4}>
                            no extractions returned.
                          </td>
                        </tr>
                      ) : null}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}

            {tab === "math" ? (
              <div className="space-y-4">
                <div>
                  <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                    cosine similarity
                  </div>
                  <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-3 text-[12px] text-[var(--text-primary)]">
                    <Latex display tex={"\\cos(\\theta)=\\frac{\\mathbf{x}\\cdot\\mathbf{y}}{\\lVert\\mathbf{x}\\rVert\\,\\lVert\\mathbf{y}\\rVert}"} />
                  </div>
                  <p className="mt-2 text-[11px] text-[var(--text-muted)]">
                    The retriever produces a vector for the span. FAISS returns concept vectors with the highest cosine similarity.
                  </p>
                </div>

                <div>
                  <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                    rerank score (within top-k)
                  </div>
                  <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-3 text-[12px] text-[var(--text-primary)]">
                    <Latex
                      display
                      tex={"s' = s + w_{lex}\\,J(\\text{span},\\text{term}) + b_{tui} + b_{group} + p(\\text{term}) + w_{rel}\\,h"}
                    />
                  </div>
                  <p className="mt-2 text-[11px] text-[var(--text-muted)]">
                    This only reorders inside a small margin near the best FAISS hit. It is designed as a tie-breaker, not a replacement for embedding similarity.
                  </p>
                </div>

                <div>
                  <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                    distillation objective (simplified)
                  </div>
                  <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-3 text-[12px] text-[var(--text-primary)]">
                    <Latex display tex={"\\mathcal{L}=\\lVert f_{student}(x) - f_{teacher}(x) \\rVert_2^2"} />
                  </div>
                  <p className="mt-2 text-[11px] text-[var(--text-muted)]">
                    The student is trained so that the vector it produces for the same span points in the same direction as the teacher.
                  </p>
                </div>
              </div>
            ) : null}
          </div>

          {api.kind === "offline" ? (
            <div className="border-t border-[var(--border)] bg-[var(--bg-surface)] px-4 py-3 text-[11px] text-[var(--text-muted)]">
              <div className="flex items-start gap-2">
                <WarningCircle size={14} weight="fill" className="mt-0.5 shrink-0" />
                <div className="min-w-0">
                  <div className="text-[11px] text-[var(--text-muted)]">
                    api not reachable. start it with{" "}
                    <span className="font-mono text-[var(--text-primary)]">python3 -m trm_umls.api</span>.
                  </div>
                  <div className="mt-1 text-[10px] text-[var(--text-faint)]">
                    this page is showing a small offline demo output.
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          {error ? (
            <div className="border-t border-[var(--border)] bg-[var(--bg-surface)] px-4 py-3 text-[11px] text-[var(--error)]">
              {error}
            </div>
          ) : null}
        </div>

        <h2>sapbert in one paragraph</h2>
        <p>
          SapBERT is a BERT-style encoder trained so that different surface forms of the same UMLS concept
          (synonyms) are embedded close together. That makes it a strong teacher for concept retrieval.
          If the teacher already clusters “htn” and “hypertension”, the student only has to learn to land in that cluster.
        </p>

        <h2>what “tiny retriever model” means here</h2>
        <p>
          It is not a generative model and it is not doing multi-step reasoning. It is a small text encoder that outputs a single vector.
          That vector is used as a key into a large index that holds the knowledge (UMLS).
        </p>

        <h2>what the system returns</h2>
        <ul className="list-disc list-inside space-y-1 text-[var(--text-body)]">
          <li>span text and character offsets</li>
          <li>cui and preferred term</li>
          <li>semantic type (tui) and semantic group</li>
          <li>similarity score + optional candidate list (debug)</li>
          <li>assertion (present / absent / possible) and subject (patient / family / other)</li>
        </ul>

        <h2>where this tends to fail</h2>
        <p>
          You will mostly see issues around span boundaries (too short, too long) and disambiguation between close concepts.
          The pipeline has guardrails (header stoplist, short-span rules, rerank margin) but it will not be perfect without a gold set.
        </p>

        <hr />

        <div className="mt-8 text-xs text-[var(--text-muted)]">
          <p className="mb-2">Citation:</p>
          <pre className="text-[10px]">
{`@misc{qassemf2026trmumls,
  author = {Qassemf, Layth M.},
  title = {trm-umls: a small embedding-distilled umls concept linker for clinical text},
  year = {2026}
}`}
          </pre>
        </div>
      </article>
    </div>
  );
}

function rowKey(r: ConceptExtraction): string {
  return `${r.start}:${r.end}:${r.cui}:${r.assertion}:${r.subject}`;
}

