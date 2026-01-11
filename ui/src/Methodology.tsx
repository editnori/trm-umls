import { useEffect, useMemo, useState } from "react";
import mermaid from "mermaid";
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

  // Initialize mermaid
  useEffect(() => {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    mermaid.initialize({
      startOnLoad: false,
      theme: isDark ? "dark" : "neutral",
      themeVariables: isDark
        ? { primaryColor: "#211d1c", primaryTextColor: "#e8e8e8", lineColor: "#505050" }
        : { primaryColor: "#f7f6f5", primaryTextColor: "#201d1d", lineColor: "#b8b6b6" },
    });
  }, []);

  // Render mermaid diagrams after content loads
  useEffect(() => {
    const nodes = document.querySelectorAll(".mermaid-src");
    nodes.forEach(async (node, i) => {
      const code = node.textContent || "";
      try {
        const { svg } = await mermaid.render(`mermaid-${i}`, code);
        const container = node.nextElementSibling;
        if (container) container.innerHTML = svg;
      } catch {
        // mermaid parse error, leave blank
      }
    });
  }, []);

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

  // Parse paper.md into sections
  const sections = useMemo(() => parsePaper(paperRaw), []);

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
      <main className="max-w-3xl mx-auto px-6 py-10">
        {sections.map((section, i) => (
          <div key={i}>
            <Section content={section} />
            
            {/* Insert interactive demo after "system overview" section */}
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
          </div>
        ))}
      </main>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Interactive Demo Component
   ───────────────────────────────────────────────────────────────────────────── */

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
    <section className="my-10 rounded border border-border bg-surface">
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

/* ─────────────────────────────────────────────────────────────────────────────
   Section Renderer
   ───────────────────────────────────────────────────────────────────────────── */

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

/* ─────────────────────────────────────────────────────────────────────────────
   Markdown Content Renderer
   ───────────────────────────────────────────────────────────────────────────── */

function MarkdownContent({ md }: { md: string }) {
  const blocks = useMemo(() => parseMarkdownBlocks(md), [md]);

  return (
    <>
      {blocks.map((block, i) => {
        if (block.type === "mermaid") {
          return (
            <div key={i} className="my-4">
              <pre className="mermaid-src hidden">{block.content}</pre>
              <div className="mermaid-render overflow-x-auto p-4 bg-surface rounded border border-border" />
            </div>
          );
        }
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
        // paragraph
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

/* ─────────────────────────────────────────────────────────────────────────────
   Parsers
   ───────────────────────────────────────────────────────────────────────────── */

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
      // Save previous section
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

  // Save last section
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

    // Math block $$
    if (line.trim() === "$$") {
      const start = i + 1;
      i++;
      while (i < lines.length && lines[i].trim() !== "$$") i++;
      blocks.push({ type: "math", content: lines.slice(start, i).join("\n") });
      i++;
      continue;
    }

    // Code block
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

    // Table (starts with |)
    if (line.startsWith("|")) {
      const start = i;
      while (i < lines.length && lines[i].startsWith("|")) i++;
      blocks.push({ type: "table", content: lines.slice(start, i).join("\n") });
      continue;
    }

    // Skip blank lines
    if (!line.trim()) {
      i++;
      continue;
    }

    // Paragraph - collect until blank line or special block
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
