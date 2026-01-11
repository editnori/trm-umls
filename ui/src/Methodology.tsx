import { useEffect, useMemo, useState } from "react";
import { extractSingle, getHealth } from "./api";
import { cx } from "./components/cx";
import { DEMO_EXTRACTIONS, DEMO_NOTE } from "./demo_data";
import type { Candidate, ConceptExtraction, ExtractOptions } from "./types";

const DEFAULT_OPTIONS: ExtractOptions = {
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

type ApiState = "checking" | "offline" | "loading" | "ready";

export default function Methodology() {
  const [api, setApi] = useState<ApiState>("checking");
  const [live] = useState(true);
  const [text, setText] = useState(DEMO_NOTE);
  const [options] = useState<ExtractOptions>(DEFAULT_OPTIONS);
  const [busy, setBusy] = useState(false);
  const [rows, setRows] = useState<ConceptExtraction[]>(DEMO_EXTRACTIONS);
  const [selected, setSelected] = useState<string | null>(null);

  // Check API health
  useEffect(() => {
    let mounted = true;
    async function check() {
      try {
        const h = await getHealth();
        if (!mounted) return;
        if (!h.ok) setApi("offline");
        else if (!h.loaded) setApi("loading");
        else setApi("ready");
      } catch {
        if (mounted) setApi("offline");
      }
    }
    void check();
    const id = setInterval(check, 5000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  // Auto-run when API ready
  useEffect(() => {
    if (api === "ready" && live) void run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  // Auto-select first row
  useEffect(() => {
    if (!selected && rows.length) setSelected(rowKey(rows[0]));
  }, [rows, selected]);

  const usingLive = live && api === "ready";
  const selectedRow = useMemo(() => rows.find(r => rowKey(r) === selected) ?? null, [rows, selected]);
  const candidates = useMemo(() => (selectedRow?.candidates ?? null) as Candidate[] | null, [selectedRow]);

  async function run() {
    setBusy(true);
    try {
      if (!usingLive) {
        setRows(DEMO_EXTRACTIONS);
        return;
      }
      const res = await extractSingle(text, options);
      setRows(res.extractions);
    } catch {
      setRows(DEMO_EXTRACTIONS);
    } finally {
      setBusy(false);
    }
  }

  const statusLabel = api === "checking" ? "checking" : api === "offline" ? "demo mode" : api === "loading" ? "loading model" : "live";

  return (
    <div className="min-h-screen bg-page">
      {/* Header */}
      <header className="flex h-9 items-center px-4 border-b border-border">
        <span className="text-[11px] text-muted font-medium">trm-umls</span>
        <span className="mx-2 text-faint">·</span>
        <span className="text-[11px] text-faint">{statusLabel}</span>
        <a href="/" className="ml-auto text-[11px] text-muted hover:text-primary">← extractor</a>
      </header>

      {/* Content */}
      <main className="max-w-2xl mx-auto px-6 py-10">
        <h1 className="text-xl font-semibold text-primary">How trm-umls works</h1>
        <p className="mt-2 text-sm text-muted">A retrieval-based concept linker for clinical text.</p>

        <hr className="my-8 border-border" />

        {/* Section 1: The Goal */}
        <section>
          <h2 className="text-sm font-semibold text-primary mb-3">What it does</h2>
          <p className="text-sm text-body leading-relaxed">
            You give it clinical text. It finds medical concepts and returns structured data: 
            the span of text, a UMLS concept ID (CUI), whether the finding is present or denied, 
            and who it refers to (patient or family).
          </p>
          <div className="mt-4 p-3 bg-surface rounded border border-border font-mono text-xs">
            <div className="text-muted mb-2">input:</div>
            <div className="text-primary">"Patient denies chest pain. History of HTN."</div>
            <div className="text-muted mt-3 mb-2">output:</div>
            <div className="text-primary">chest pain → C0008031 (ABSENT)</div>
            <div className="text-primary">HTN → C0020538 (PRESENT)</div>
          </div>
        </section>

        <hr className="my-8 border-border" />

        {/* Section 2: The Approach */}
        <section>
          <h2 className="text-sm font-semibold text-primary mb-3">How it works</h2>
          <p className="text-sm text-body leading-relaxed mb-4">
            The system uses <strong>retrieval</strong>, not generation. Instead of training a huge model 
            to "know" all of UMLS, we:
          </p>
          <ol className="text-sm text-body leading-relaxed space-y-2 ml-4">
            <li><span className="text-muted">1.</span> Embed all 1.16M UMLS concepts once using SapBERT (a biomedical encoder)</li>
            <li><span className="text-muted">2.</span> Build a FAISS index for fast nearest-neighbor search</li>
            <li><span className="text-muted">3.</span> Train a tiny model (9.8M params) to project clinical text into the same space</li>
          </ol>
          <p className="text-sm text-body leading-relaxed mt-4">
            At runtime, we embed the text span, search FAISS for the closest concept, and return it.
            The small model only needs to <em>point</em> to the right neighborhood—it doesn't store medical knowledge itself.
          </p>
        </section>

        <hr className="my-8 border-border" />

        {/* Section 3: Try It */}
        <section>
          <h2 className="text-sm font-semibold text-primary mb-3">Try it</h2>
          <p className="text-sm text-body leading-relaxed mb-4">
            {api === "offline" 
              ? "The API is offline, so this shows demo output. Start the API to see live results."
              : "Paste text below and click Run to see extractions."}
          </p>

          <div className="rounded border border-border overflow-hidden">
            {/* Input */}
            <div className="p-3 bg-elevated">
              <div className="text-[10px] font-semibold uppercase tracking-wide text-muted mb-2">note text</div>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={3}
                className="w-full resize-none bg-transparent text-sm text-primary font-mono outline-none"
                spellCheck={false}
              />
              <div className="flex items-center gap-2 mt-2">
                <button
                  onClick={() => void run()}
                  disabled={busy}
                  className="px-3 py-1 text-xs font-medium rounded bg-surface border border-border text-primary hover:bg-hover disabled:opacity-50"
                >
                  {busy ? "running…" : "run"}
                </button>
                <span className="text-[10px] text-faint">
                  {usingLive ? "live" : "demo"} · threshold {options.threshold}
                </span>
              </div>
            </div>

            {/* Results */}
            <div className="border-t border-border p-3 bg-surface">
              <div className="text-[10px] font-semibold uppercase tracking-wide text-muted mb-2">
                extractions ({rows.length})
              </div>
              <div className="space-y-1">
                {rows.map((r) => {
                  const key = rowKey(r);
                  const isSelected = selected === key;
                  return (
                    <button
                      key={key}
                      onClick={() => setSelected(key)}
                      className={cx(
                        "w-full text-left px-2 py-1.5 rounded text-xs font-mono",
                        isSelected ? "bg-elevated border border-border-strong" : "hover:bg-hover"
                      )}
                    >
                      <span className="text-primary">{r.text}</span>
                      <span className="text-muted ml-2">→ {r.cui}</span>
                      <span className={cx("ml-2", r.assertion === "ABSENT" ? "text-error" : "text-success")}>
                        {r.assertion}
                      </span>
                      {r.subject === "FAMILY" && <span className="ml-2 text-warning">FAMILY</span>}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </section>

        <hr className="my-8 border-border" />

        {/* Section 4: Retrieval Detail */}
        <section>
          <h2 className="text-sm font-semibold text-primary mb-3">Inside the retrieval</h2>
          <p className="text-sm text-body leading-relaxed mb-4">
            When you select an extraction above, you can see the candidate concepts that FAISS returned. 
            The system picks the best one using a combination of embedding similarity and light reranking.
          </p>

          {selectedRow && (
            <div className="rounded border border-border overflow-hidden">
              <div className="p-3 bg-elevated">
                <div className="text-xs font-mono text-primary">{selectedRow.text}</div>
                <div className="text-[10px] text-muted mt-1">
                  normalized: <span className="font-mono">{selectedRow.normalized_text}</span>
                </div>
              </div>

              {candidates && candidates.length > 0 ? (
                <div className="border-t border-border">
                  <table className="w-full text-[11px]">
                    <thead className="bg-surface text-[10px] uppercase tracking-wide text-muted">
                      <tr>
                        <th className="px-3 py-2 text-left">#</th>
                        <th className="px-3 py-2 text-left">cui</th>
                        <th className="px-3 py-2 text-left">term</th>
                        <th className="px-3 py-2 text-right">sim</th>
                        <th className="px-3 py-2 text-right">final</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono">
                      {candidates.slice(0, 5).map((c, i) => (
                        <tr key={c.cui} className={cx("border-t border-border", i === 0 && "bg-elevated")}>
                          <td className="px-3 py-2 text-muted">{c.rank + 1}</td>
                          <td className="px-3 py-2 text-muted">{c.cui}</td>
                          <td className="px-3 py-2 text-primary font-sans">{c.preferred_term}</td>
                          <td className="px-3 py-2 text-right text-muted">{c.score.toFixed(2)}</td>
                          <td className="px-3 py-2 text-right text-primary">{c.rerank_score.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="p-3 border-t border-border text-xs text-muted">
                  No candidate list available for this span.
                </div>
              )}
            </div>
          )}

          <p className="text-sm text-body leading-relaxed mt-4">
            The <strong>sim</strong> column is cosine similarity from FAISS. The <strong>final</strong> score 
            adds small adjustments for lexical overlap and semantic type preferences. These are tie-breakers—a 
            clearly better embedding match always wins.
          </p>
        </section>

        <hr className="my-8 border-border" />

        {/* Section 5: Key Numbers */}
        <section>
          <h2 className="text-sm font-semibold text-primary mb-3">Key numbers</h2>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-3 bg-surface rounded border border-border">
              <div className="text-muted">model size</div>
              <div className="text-primary font-mono mt-1">9.8M params</div>
            </div>
            <div className="p-3 bg-surface rounded border border-border">
              <div className="text-muted">UMLS concepts</div>
              <div className="text-primary font-mono mt-1">1,164,238</div>
            </div>
            <div className="p-3 bg-surface rounded border border-border">
              <div className="text-muted">embedding dim</div>
              <div className="text-primary font-mono mt-1">768</div>
            </div>
            <div className="p-3 bg-surface rounded border border-border">
              <div className="text-muted">throughput</div>
              <div className="text-primary font-mono mt-1">~130 rows/sec</div>
            </div>
          </div>
        </section>

        <hr className="my-8 border-border" />

        {/* Section 6: Limitations */}
        <section>
          <h2 className="text-sm font-semibold text-primary mb-3">Where it struggles</h2>
          <ul className="text-sm text-body leading-relaxed space-y-2">
            <li><span className="text-muted">•</span> <strong>Span boundaries</strong> — sometimes extracts too much or too little text</li>
            <li><span className="text-muted">•</span> <strong>Disambiguation</strong> — "BUN" could be blood urea nitrogen or a bread roll</li>
            <li><span className="text-muted">•</span> <strong>Negation scope</strong> — "no HTN, DM, or CAD" may only negate the first item</li>
            <li><span className="text-muted">•</span> <strong>No gold set</strong> — accuracy metrics require human-labeled data we don't have yet</li>
          </ul>
        </section>

        <hr className="my-8 border-border" />

        {/* Section 7: Learn More */}
        <section>
          <h2 className="text-sm font-semibold text-primary mb-3">Learn more</h2>
          <p className="text-sm text-body leading-relaxed">
            The full technical details are in <a href="https://github.com/editnori/trm-umls" className="text-primary underline">the repository README</a>.
            It covers the teacher-student distillation, FAISS indexing, reranking formula, and training process.
          </p>
        </section>

        {/* Citation */}
        <div className="mt-12 pt-6 border-t border-border">
          <div className="text-[10px] text-muted uppercase tracking-wide mb-2">citation</div>
          <pre className="text-[10px] text-muted font-mono leading-relaxed">
{`@misc{qassemf2026trmumls,
  author = {Qassemf, Layth M.},
  title = {trm-umls: embedding-distilled UMLS linker},
  year = {2026}
}`}
          </pre>
        </div>
      </main>
    </div>
  );
}

function rowKey(r: ConceptExtraction): string {
  return `${r.start}:${r.end}:${r.cui}`;
}
