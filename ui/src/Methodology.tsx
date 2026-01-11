import { useEffect, useMemo, useState } from "react";
import { extractSingle, getHealth } from "./api";
import { cx } from "./components/cx";
import { DEMO_EXTRACTIONS, DEMO_NOTE } from "./demo_data";
import type { Candidate, ConceptExtraction, ExtractOptions } from "./types";
import paperRaw from "../../trm_umls/paper.md?raw";

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

  useEffect(() => {
    if (api === "ready" && live) void run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

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
  const sections = useMemo(() => parsePaper(paperRaw), []);

  return (
    <div className="min-h-screen bg-page">
      <header className="flex h-9 items-center px-4 border-b border-border">
        <span className="text-[11px] text-muted font-medium">trm-umls</span>
        <span className="mx-2 text-faint">·</span>
        <span className="text-[11px] text-faint">{statusLabel}</span>
        <a href="/" className="ml-auto text-[11px] text-muted hover:text-primary">← extractor</a>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-10">
        {sections.map((section, i) => (
          <div key={i}>
            <Section content={section} />
            
            {/* Interactive demo after system overview */}
            {section.id === "2-system-overview" && (
              <InteractiveDemo
                api={api}
                statusLabel={statusLabel}
                usingLive={usingLive}
                text={text}
                setText={setText}
                options={options}
                busy={busy}
                run={run}
                rows={rows}
                selected={selected}
                setSelected={setSelected}
                selectedRow={selectedRow}
                candidates={candidates}
              />
            )}

            {/* Embedding space visual after "models and indexing" intro */}
            {section.id === "3-models-and-indexing" && <EmbeddingSpaceVisual />}

            {/* Recursive model visual */}
            {section.id === "3-models-and-indexing" && <RecursiveModelVisual />}

            {/* Contrastive loss visual after training objective */}
            {section.id === "4-training-objective" && <ContrastiveLossVisual />}

            {/* Rerank visual after linking and reranking */}
            {section.id === "6-linking-and-reranking" && <RerankVisual />}
          </div>
        ))}
      </main>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   VISUAL: Embedding Space
   ═══════════════════════════════════════════════════════════════════════════ */

function EmbeddingSpaceVisual() {
  const [hovered, setHovered] = useState<string | null>(null);

  const clusters = [
    { id: "htn", label: "hypertension cluster", x: 20, y: 25, terms: ["hypertension", "HTN", "high blood pressure", "elevated BP"] },
    { id: "dm", label: "diabetes cluster", x: 70, y: 60, terms: ["diabetes", "DM", "diabetes mellitus", "type 2 DM"] },
    { id: "hypotension", label: "hypotension cluster", x: 75, y: 20, terms: ["hypotension", "low BP"] },
  ];

  const activeCluster = clusters.find(c => c.id === hovered);

  return (
    <div className="my-8 rounded border border-border bg-surface p-4">
      <div className="text-[10px] font-semibold uppercase tracking-wide text-muted mb-3">
        embedding space: how terms cluster
      </div>
      <div className="flex gap-4">
        <div className="relative w-64 h-48 bg-elevated rounded border border-border">
          {/* Grid lines */}
          <div className="absolute inset-0 opacity-20">
            {[25, 50, 75].map(p => (
              <div key={`h${p}`} className="absolute w-full border-t border-border" style={{ top: `${p}%` }} />
            ))}
            {[25, 50, 75].map(p => (
              <div key={`v${p}`} className="absolute h-full border-l border-border" style={{ left: `${p}%` }} />
            ))}
          </div>
          {/* Clusters */}
          {clusters.map(c => (
            <button
              key={c.id}
              className={cx(
                "absolute w-16 h-10 rounded border text-[9px] font-mono transition-all",
                hovered === c.id
                  ? "bg-surface border-border-strong text-primary z-10 scale-110"
                  : "bg-elevated border-border text-muted hover:border-border-strong"
              )}
              style={{ left: `${c.x}%`, top: `${c.y}%`, transform: "translate(-50%, -50%)" }}
              onMouseEnter={() => setHovered(c.id)}
              onMouseLeave={() => setHovered(null)}
            >
              {c.id}
            </button>
          ))}
          <div className="absolute bottom-1 right-2 text-[8px] text-faint">768 dimensions →</div>
        </div>
        <div className="flex-1 text-xs text-body">
          {activeCluster ? (
            <>
              <div className="text-primary font-medium mb-2">{activeCluster.label}</div>
              <div className="space-y-1">
                {activeCluster.terms.map(t => (
                  <div key={t} className="font-mono text-[11px] text-muted">"{t}"</div>
                ))}
              </div>
              <div className="mt-3 text-[10px] text-faint">
                all map to the same CUI because they cluster together
              </div>
            </>
          ) : (
            <div className="text-muted">
              hover a cluster to see which terms land nearby in embedding space.
              <br /><br />
              <span className="text-faint">
                similar meanings → similar vectors → same region
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   VISUAL: Recursive Model
   ═══════════════════════════════════════════════════════════════════════════ */

function RecursiveModelVisual() {
  const [step, setStep] = useState(0);
  const maxSteps = 12;

  return (
    <div className="my-8 rounded border border-border bg-surface p-4">
      <div className="text-[10px] font-semibold uppercase tracking-wide text-muted mb-3">
        recursive computation: same weights, multiple passes
      </div>
      <div className="flex items-start gap-6">
        <div className="flex flex-col items-center gap-2">
          <div className="px-3 py-2 bg-elevated rounded border border-border text-xs font-mono">
            "chest pain"
          </div>
          <div className="text-[10px] text-faint">↓</div>
          <div className="relative px-4 py-3 bg-elevated rounded border border-border-strong">
            <div className="text-[10px] text-muted mb-1">TRM Block</div>
            <div className="text-xs font-mono text-primary">9.8M params</div>
            <div className="absolute -right-2 top-1/2 -translate-y-1/2 text-[10px] text-faint">
              ↺ {step + 1}x
            </div>
          </div>
          <div className="text-[10px] text-faint">↓</div>
          <div className="px-3 py-2 bg-elevated rounded border border-border text-xs font-mono">
            768-d vector
          </div>
        </div>
        <div className="flex-1">
          <div className="text-xs text-body mb-3">
            instead of 12 separate layers with 12 separate parameter sets, the encoder runs <strong className="text-primary">one layer repeatedly</strong>.
          </div>
          <div className="flex items-center gap-2 mb-3">
            <input
              type="range"
              min={0}
              max={maxSteps - 1}
              value={step}
              onChange={e => setStep(Number(e.target.value))}
              className="flex-1"
            />
            <span className="text-xs font-mono text-muted w-12">pass {step + 1}</span>
          </div>
          <div className="flex gap-1">
            {Array.from({ length: maxSteps }).map((_, i) => (
              <div
                key={i}
                className={cx(
                  "h-2 flex-1 rounded-sm transition-colors",
                  i <= step ? "bg-primary" : "bg-border"
                )}
              />
            ))}
          </div>
          <div className="mt-3 text-[10px] text-faint">
            3 high-level × 4 low-level = 12 effective passes. same weights, deeper reasoning.
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   VISUAL: Contrastive Loss
   ═══════════════════════════════════════════════════════════════════════════ */

function ContrastiveLossVisual() {
  const [activeSpan, setActiveSpan] = useState(0);

  const examples = [
    { span: "high blood pressure", correct: "C0020538", correctName: "Hypertensive disease", distractors: ["C0011849", "C0232289"] },
    { span: "type 2 diabetes", correct: "C0011860", correctName: "Type 2 DM", distractors: ["C0020538", "C0006142"] },
    { span: "chest discomfort", correct: "C0232289", correctName: "Chest discomfort", distractors: ["C0011860", "C0020538"] },
  ];

  const ex = examples[activeSpan];

  return (
    <div className="my-8 rounded border border-border bg-surface p-4">
      <div className="text-[10px] font-semibold uppercase tracking-wide text-muted mb-3">
        contrastive loss: learn to distinguish
      </div>
      <div className="grid grid-cols-3 gap-2 mb-4">
        {examples.map((e, i) => (
          <button
            key={i}
            onClick={() => setActiveSpan(i)}
            className={cx(
              "px-2 py-1.5 rounded text-[11px] font-mono text-left",
              activeSpan === i ? "bg-elevated border border-border-strong text-primary" : "text-muted hover:text-primary"
            )}
          >
            "{e.span}"
          </button>
        ))}
      </div>
      <div className="flex gap-4">
        <div className="flex-1">
          <div className="text-[10px] text-muted mb-2">embedding similarity</div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-20 text-[10px] font-mono text-muted truncate">{ex.correct}</div>
              <div className="flex-1 h-4 bg-elevated rounded overflow-hidden">
                <div className="h-full bg-[var(--success)]" style={{ width: "92%" }} />
              </div>
              <div className="w-10 text-[10px] font-mono text-[var(--success)]">0.92</div>
              <div className="text-[10px] text-[var(--success)]">✓ correct</div>
            </div>
            {ex.distractors.map((d, i) => (
              <div key={d} className="flex items-center gap-2">
                <div className="w-20 text-[10px] font-mono text-muted truncate">{d}</div>
                <div className="flex-1 h-4 bg-elevated rounded overflow-hidden">
                  <div className="h-full bg-border" style={{ width: `${30 - i * 8}%` }} />
                </div>
                <div className="w-10 text-[10px] font-mono text-faint">{(0.3 - i * 0.08).toFixed(2)}</div>
                <div className="text-[10px] text-faint">distractor</div>
              </div>
            ))}
          </div>
        </div>
        <div className="w-px bg-border" />
        <div className="w-48">
          <div className="text-[10px] text-muted mb-2">the loss says:</div>
          <div className="text-xs text-body">
            push <span className="font-mono text-primary">"{ex.span}"</span> 
            <br />close to <span className="font-mono text-[var(--success)]">{ex.correctName}</span>
            <br />far from distractors
          </div>
          <div className="mt-3 p-2 bg-elevated rounded border border-border font-mono text-[10px] text-muted">
            sim(span, correct) → high<br />
            sim(span, others) → low
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   VISUAL: Reranking
   ═══════════════════════════════════════════════════════════════════════════ */

function RerankVisual() {
  const [lexWeight, setLexWeight] = useState(0.3);

  const candidates = [
    { cui: "C0020538", term: "Hypertensive disease", sim: 0.89, lex: 0.33, bias: 0.01 },
    { cui: "C0235222", term: "Hypertensive heart disease", sim: 0.87, lex: 0.25, bias: 0.01 },
    { cui: "C0020545", term: "Hypertension, Portal", sim: 0.85, lex: 0.50, bias: 0.006 },
  ];

  const scored = candidates.map(c => ({
    ...c,
    final: c.sim + lexWeight * c.lex + c.bias,
  })).sort((a, b) => b.final - a.final);

  return (
    <div className="my-8 rounded border border-border bg-surface p-4">
      <div className="text-[10px] font-semibold uppercase tracking-wide text-muted mb-3">
        reranking: combining signals for "hypertension"
      </div>
      <div className="flex gap-4">
        <div className="flex-1">
          <div className="overflow-hidden rounded border border-border">
            <table className="w-full text-[11px]">
              <thead className="bg-elevated text-[10px] uppercase tracking-wide text-muted">
                <tr>
                  <th className="px-2 py-1.5 text-left">term</th>
                  <th className="px-2 py-1.5 text-right">sim</th>
                  <th className="px-2 py-1.5 text-right">lex</th>
                  <th className="px-2 py-1.5 text-right">bias</th>
                  <th className="px-2 py-1.5 text-right">final</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {scored.map((c, i) => (
                  <tr key={c.cui} className={cx("border-t border-border", i === 0 && "bg-elevated")}>
                    <td className="px-2 py-1.5 text-primary font-sans">{c.term}</td>
                    <td className="px-2 py-1.5 text-right text-muted">{c.sim.toFixed(2)}</td>
                    <td className="px-2 py-1.5 text-right text-muted">{c.lex.toFixed(2)}</td>
                    <td className="px-2 py-1.5 text-right text-faint">{c.bias.toFixed(3)}</td>
                    <td className="px-2 py-1.5 text-right text-primary">{c.final.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="w-48">
          <div className="text-[10px] text-muted mb-2">formula</div>
          <div className="p-2 bg-elevated rounded border border-border font-mono text-[10px] text-body mb-3">
            final = sim<br />
            &nbsp;&nbsp;+ <span className="text-primary">{lexWeight.toFixed(1)}</span> × lex<br />
            &nbsp;&nbsp;+ bias
          </div>
          <div className="text-[10px] text-muted mb-1">lexical weight</div>
          <input
            type="range"
            min={0}
            max={0.8}
            step={0.1}
            value={lexWeight}
            onChange={e => setLexWeight(Number(e.target.value))}
            className="w-full"
          />
          <div className="text-[10px] text-faint mt-2">
            adjust to see how lexical overlap affects ranking
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   INTERACTIVE DEMO
   ═══════════════════════════════════════════════════════════════════════════ */

interface DemoProps {
  api: ApiState;
  statusLabel: string;
  usingLive: boolean;
  text: string;
  setText: (t: string) => void;
  options: ExtractOptions;
  busy: boolean;
  run: () => void;
  rows: ConceptExtraction[];
  selected: string | null;
  setSelected: (s: string | null) => void;
  selectedRow: ConceptExtraction | null;
  candidates: Candidate[] | null;
}

function InteractiveDemo({
  api, statusLabel, usingLive, text, setText, options, busy, run,
  rows, selected, setSelected, selectedRow, candidates
}: DemoProps) {
  return (
    <section className="my-8 rounded border border-border bg-surface">
      <div className="flex items-center gap-2 border-b border-border px-4 py-2 text-[10px] uppercase tracking-wide text-muted">
        try it now
        <span className="text-faint">·</span>
        <span className="normal-case text-[11px]">{statusLabel}</span>
        <button
          onClick={run}
          disabled={busy}
          className="ml-auto rounded px-2 py-1 text-[10px] text-muted hover:text-primary disabled:opacity-50"
        >
          {busy ? "running…" : "run"}
        </button>
      </div>

      <div className="p-4 bg-elevated">
        <div className="text-[10px] font-semibold uppercase tracking-wide text-muted mb-2">note text</div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
          className="w-full resize-none bg-transparent text-sm text-primary font-mono outline-none"
          spellCheck={false}
        />
        <div className="mt-2 text-[10px] text-faint">
          {usingLive ? "live api" : "demo output"} · threshold {options.threshold}
        </div>
      </div>

      <div className="border-t border-border p-4 bg-surface">
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
                <span className={cx("ml-2", r.assertion === "ABSENT" ? "text-[var(--error)]" : "text-[var(--success)]")}>
                  {r.assertion}
                </span>
                {r.subject === "FAMILY" && <span className="ml-2 text-[var(--warning)]">FAMILY</span>}
              </button>
            );
          })}
        </div>
      </div>

      {selectedRow && candidates && candidates.length > 0 && (
        <div className="border-t border-border">
          <div className="px-4 py-2 text-[10px] font-semibold uppercase tracking-wide text-muted bg-surface">
            faiss candidates for "{selectedRow.text}"
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]">
              <thead className="bg-surface text-[10px] uppercase tracking-wide text-muted">
                <tr>
                  <th className="px-3 py-2 text-left">#</th>
                  <th className="px-3 py-2 text-left">cui</th>
                  <th className="px-3 py-2 text-left">term</th>
                  <th className="px-3 py-2 text-right">sim</th>
                  <th className="px-3 py-2 text-right">rerank</th>
                </tr>
              </thead>
              <tbody className="font-mono bg-elevated">
                {candidates.slice(0, 5).map((c, i) => (
                  <tr key={c.cui} className={cx("border-t border-border", i === 0 && "bg-surface")}>
                    <td className="px-3 py-2 text-muted">{c.rank + 1}</td>
                    <td className="px-3 py-2 text-muted">{c.cui}</td>
                    <td className="px-3 py-2 text-primary font-sans">{c.preferred_term}</td>
                    <td className="px-3 py-2 text-right text-muted">{c.score.toFixed(3)}</td>
                    <td className="px-3 py-2 text-right text-primary">{c.rerank_score.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {api === "offline" && (
        <div className="border-t border-border bg-surface px-4 py-3 text-[11px] text-muted">
          api offline. start with <span className="font-mono text-primary">python3 -m trm_umls.api</span>
        </div>
      )}
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION & MARKDOWN RENDERING
   ═══════════════════════════════════════════════════════════════════════════ */

interface ParsedSection {
  id: string;
  level: number;
  title: string;
  content: string;
}

function Section({ content }: { content: ParsedSection }) {
  return (
    <section id={content.id} className="mb-8">
      {content.level === 1 && (
        <h1 className="text-lg font-semibold text-primary mb-2">{content.title}</h1>
      )}
      {content.level === 2 && (
        <>
          <hr className="border-border mb-6" />
          <h2 className="text-sm font-semibold text-primary mb-3">{content.title}</h2>
        </>
      )}
      {content.level === 3 && (
        <h3 className="text-xs font-semibold text-primary mt-4 mb-2">{content.title}</h3>
      )}
      <div className="prose-content">
        <MarkdownContent md={content.content} />
      </div>
    </section>
  );
}

function MarkdownContent({ md }: { md: string }) {
  const blocks = useMemo(() => parseMarkdownBlocks(md), [md]);

  return (
    <>
      {blocks.map((block, i) => {
        // Skip mermaid blocks - we replace with custom visuals
        if (block.type === "mermaid") return null;
        // Skip ASCII art blocks (768-dimensional space)
        if (block.type === "code" && block.content.includes("768-dimensional")) return null;
        
        if (block.type === "code") {
          return (
            <pre key={i} className="my-4 p-3 bg-surface rounded border border-border overflow-x-auto text-xs font-mono text-body">
              {block.content}
            </pre>
          );
        }
        if (block.type === "table") {
          return <MarkdownTable key={i} content={block.content} />;
        }
        if (block.type === "math") {
          return (
            <div key={i} className="my-4 p-3 bg-surface rounded border border-border overflow-x-auto text-xs font-mono text-body">
              {block.content}
            </div>
          );
        }
        // Skip "---" dividers and "The diagram below" references
        if (block.content.trim() === "---") return null;
        if (block.content.includes("diagram below") || block.content.includes("diagram above")) return null;
        
        return (
          <p key={i} className="text-sm text-body leading-relaxed mb-3" dangerouslySetInnerHTML={{ __html: inlineMarkdown(block.content) }} />
        );
      })}
    </>
  );
}

function MarkdownTable({ content }: { content: string }) {
  const lines = content.trim().split("\n").filter(l => l.trim());
  if (lines.length < 2) return null;

  const parseRow = (line: string) =>
    line.split("|").map(c => c.trim()).filter(c => c);

  const headers = parseRow(lines[0]);
  const rows = lines.slice(2).map(parseRow);

  return (
    <div className="my-4 overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="border-b border-border">
            {headers.map((h, i) => (
              <th key={i} className="px-3 py-2 text-left text-muted font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className="border-b border-border">
              {row.map((cell, ci) => (
                <td key={ci} className="px-3 py-2 text-body" dangerouslySetInnerHTML={{ __html: inlineMarkdown(cell) }} />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   PARSERS
   ═══════════════════════════════════════════════════════════════════════════ */

function parsePaper(raw: string): ParsedSection[] {
  const sections: ParsedSection[] = [];
  const lines = raw.split("\n");

  let currentSection: ParsedSection | null = null;
  let buffer: string[] = [];

  for (const line of lines) {
    const h1 = line.match(/^# (.+)$/);
    const h2 = line.match(/^## (.+)$/);
    const h3 = line.match(/^### (.+)$/);

    if (h1 || h2 || h3) {
      if (currentSection) {
        currentSection.content = buffer.join("\n").trim();
        sections.push(currentSection);
      }

      const title = (h1?.[1] || h2?.[1] || h3?.[1] || "").trim();
      const level = h1 ? 1 : h2 ? 2 : 3;
      const id = title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

      currentSection = { id, level, title, content: "" };
      buffer = [];
    } else {
      buffer.push(line);
    }
  }

  if (currentSection) {
    currentSection.content = buffer.join("\n").trim();
    sections.push(currentSection);
  }

  return sections;
}

type BlockType = "paragraph" | "code" | "mermaid" | "table" | "math";

function parseMarkdownBlocks(md: string): { type: BlockType; content: string }[] {
  const blocks: { type: BlockType; content: string }[] = [];
  const lines = md.split("\n");

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === "$$") {
      const start = i + 1;
      i++;
      while (i < lines.length && lines[i].trim() !== "$$") i++;
      blocks.push({ type: "math", content: lines.slice(start, i).join("\n") });
      i++;
      continue;
    }

    const codeMatch = line.match(/^```(\w*)$/);
    if (codeMatch) {
      const lang = codeMatch[1];
      const start = i + 1;
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) i++;
      const content = lines.slice(start, i).join("\n");
      blocks.push({ type: lang === "mermaid" ? "mermaid" : "code", content });
      i++;
      continue;
    }

    if (line.startsWith("|")) {
      const start = i;
      while (i < lines.length && lines[i].startsWith("|")) i++;
      blocks.push({ type: "table", content: lines.slice(start, i).join("\n") });
      continue;
    }

    if (!line.trim()) {
      i++;
      continue;
    }

    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].startsWith("```") &&
      !lines[i].startsWith("|") &&
      lines[i].trim() !== "$$" &&
      !lines[i].match(/^#{1,3} /)
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    if (paraLines.length) {
      blocks.push({ type: "paragraph", content: paraLines.join(" ") });
    }
  }

  return blocks;
}

function inlineMarkdown(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong class="text-primary font-medium">$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 bg-surface rounded text-[11px] font-mono">$1</code>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-primary underline">$1</a>')
    .replace(/\$([^$]+)\$/g, '<code class="px-1 py-0.5 bg-surface rounded text-[11px] font-mono">$1</code>');
}

function rowKey(r: ConceptExtraction): string {
  return `${r.start}:${r.end}:${r.cui}`;
}
