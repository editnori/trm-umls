import { useState } from "react";
import { cx } from "./components/cx";

/**
 * Methodology page - readable article format like OpenCode docs
 * Interactive elements woven into prose, not replacing it
 */

export default function Methodology() {
  const [activeExample, setActiveExample] = useState<"input" | "spans" | "output">("input");

  return (
    <div className="min-h-screen bg-[var(--bg-page)]">
      {/* Header */}
      <header className="border-b border-[var(--border)] px-6 py-3">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <span className="text-[11px] text-[var(--text-muted)]">trm-umls</span>
          <a
            href="/"
            className="text-[11px] text-[var(--text-muted)] hover:text-[var(--text-primary)]"
          >
            ← extractor
          </a>
        </div>
      </header>

      {/* Article */}
      <article className="max-w-3xl mx-auto px-6 py-12 prose">
        <h1 className="text-2xl font-semibold text-[var(--text-primary)] mb-2">
          How trm-umls works
        </h1>
        <p className="text-[var(--text-muted)] text-sm mb-8">
          A small embedding-distilled UMLS concept linker for clinical text.
        </p>

        <hr />

        <h2>The problem</h2>
        <p>
          Clinical notes are full of medical terms, abbreviations, and jargon. "HTN", "DM", "COPD" 
          all mean something specific, but they need to be linked to standardized concepts 
          for downstream analytics, billing, or clinical decision support.
        </p>
        <p>
          UMLS contains over 1 million medical concepts. The challenge is matching messy 
          clinical text to the right concept quickly and accurately.
        </p>

        <h2>The approach</h2>
        <p>
          Instead of building a massive model that "knows" UMLS, we took a different path:
        </p>
        <blockquote>
          Train a small model to produce embeddings in the same space as a strong teacher. 
          Let nearest-neighbor search do the concept lookup.
        </blockquote>
        <p>
          The teacher is <code>SapBERT</code>, a biomedical embedding model. We embedded all 
          1.16 million UMLS concepts once, built a FAISS index, then trained a tiny student 
          model (9.8M params vs SapBERT's 110M) to project clinical text into that same space.
        </p>

        <h2>Step by step</h2>
        <p>
          Here's what happens when you paste a note. Try clicking through the stages:
        </p>

        {/* Interactive example */}
        <div className="my-6 border border-[var(--border)] rounded overflow-hidden">
          <div className="flex border-b border-[var(--border)] bg-[var(--bg-surface)]">
            {(["input", "spans", "output"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveExample(tab)}
                className={cx(
                  "px-4 py-2 text-xs font-medium transition-colors",
                  activeExample === tab
                    ? "text-[var(--text-primary)] bg-[var(--bg-elevated)]"
                    : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                )}
              >
                {tab === "input" && "1. Input"}
                {tab === "spans" && "2. Spans"}
                {tab === "output" && "3. Output"}
              </button>
            ))}
          </div>
          <div className="p-4 bg-[var(--bg-elevated)]">
            {activeExample === "input" && (
              <div>
                <p className="text-xs text-[var(--text-muted)] mb-2">Raw clinical text:</p>
                <pre className="text-sm font-mono text-[var(--text-body)] whitespace-pre-wrap">
{`Assessment: hypertension, diabetes mellitus, COPD.
Denies chest pain or shortness of breath at rest.`}
                </pre>
              </div>
            )}
            {activeExample === "spans" && (
              <div>
                <p className="text-xs text-[var(--text-muted)] mb-2">
                  NER teachers propose candidate spans:
                </p>
                <div className="space-y-1">
                  {["hypertension", "diabetes mellitus", "COPD", "chest pain", "shortness of breath at rest"].map((span) => (
                    <span
                      key={span}
                      className="inline-block mr-2 px-2 py-1 bg-[var(--bg-surface)] rounded text-xs font-mono"
                    >
                      {span}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {activeExample === "output" && (
              <div>
                <p className="text-xs text-[var(--text-muted)] mb-2">
                  Structured extractions with CUIs and assertions:
                </p>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-[var(--border)]">
                      <th className="text-left py-1 font-medium text-[var(--text-muted)]">span</th>
                      <th className="text-left py-1 font-medium text-[var(--text-muted)]">CUI</th>
                      <th className="text-left py-1 font-medium text-[var(--text-muted)]">assertion</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono">
                    <tr><td className="py-1">hypertension</td><td className="text-[var(--text-muted)]">C3280772</td><td className="text-[#166534]">PRESENT</td></tr>
                    <tr><td className="py-1">diabetes mellitus</td><td className="text-[var(--text-muted)]">C0011849</td><td className="text-[#166534]">PRESENT</td></tr>
                    <tr><td className="py-1">chest pain</td><td className="text-[var(--text-muted)]">C0008031</td><td className="text-[#b91c1c]">ABSENT</td></tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        <p>
          The key insight: <strong>"HTN"</strong>, <strong>"hypertension"</strong>, and{" "}
          <strong>"high blood pressure"</strong> all land near the same point in embedding space. 
          FAISS finds the nearest UMLS concept vector, and that's the linked CUI.
        </p>

        <h2>The embedding space</h2>
        <p>
          Think of it as a 768-dimensional room where every medical concept has a location. 
          Similar meanings cluster together:
        </p>

        <div className="my-6 p-4 bg-[var(--bg-surface)] rounded border border-[var(--border)] font-mono text-xs">
          <div className="text-[var(--text-muted)] mb-3">768-dimensional embedding space</div>
          <div className="space-y-3">
            <div>
              <span className="text-[#b45309]">Hypertension cluster:</span>
              <span className="text-[var(--text-body)] ml-2">HTN · hypertension · high blood pressure · elevated BP</span>
            </div>
            <div>
              <span className="text-[#166534]">Diabetes cluster:</span>
              <span className="text-[var(--text-body)] ml-2">DM · diabetes · diabetes mellitus · type 2 DM</span>
            </div>
            <div>
              <span className="text-[#0d7377]">Pain cluster:</span>
              <span className="text-[var(--text-body)] ml-2">chest pain · CP · angina · chest discomfort</span>
            </div>
          </div>
        </div>

        <p>
          When a clinician writes "htn" in a note, the student model projects it near the 
          hypertension cluster, and FAISS returns C0020538.
        </p>

        <h2>Why this works</h2>
        <p>
          The student model doesn't need to "know" UMLS. It only needs to project text into 
          the right neighborhood. This is why a 9.8M parameter model can match a 110M 
          parameter teacher—it's not defining the space, just learning to navigate it.
        </p>

        <h3>Teacher validation</h3>
        <p>
          We tested the teacher space (SapBERT) on 20,000 synonym queries:
        </p>
        <table>
          <thead>
            <tr>
              <th>metric</th>
              <th>SapBERT</th>
              <th>BGE-M3</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>hit@1</td>
              <td><strong>95.4%</strong></td>
              <td>88.5%</td>
            </tr>
            <tr>
              <td>hit@5</td>
              <td><strong>97.5%</strong></td>
              <td>92.6%</td>
            </tr>
          </tbody>
        </table>
        <p>
          This ceiling bounds what the student can achieve. When the teacher gets it right 
          95% of the time, that's the best case for distillation.
        </p>

        <h2>Reranking</h2>
        <p>
          Sometimes FAISS returns multiple close candidates. "Hypertension" could match 
          several CUIs: general hypertension, pulmonary hypertension, portal hypertension.
        </p>
        <p>
          Reranking adjusts scores using:
        </p>
        <ul className="list-disc list-inside space-y-1 text-[var(--text-body)]">
          <li>Lexical overlap (does the preferred term match the span?)</li>
          <li>Semantic group bias (disorders get a small boost over abstract concepts)</li>
          <li>Variant penalties (CTCAE-specific variants get penalized)</li>
        </ul>
        <p>
          These are tie-breakers, not overrides. A clearly better embedding match always wins.
        </p>

        <h2>Assertion detection</h2>
        <p>
          Linking "chest pain" to C0008031 isn't enough. The note says{" "}
          <em>"denies chest pain"</em>—that's an <strong>absent</strong> finding, not present.
        </p>
        <p>
          A small classification head predicts assertion (PRESENT / ABSENT / POSSIBLE) and 
          subject (PATIENT / FAMILY / OTHER). Rule-based overrides catch common patterns 
          like "denies", "no history of", "family history of".
        </p>

        <h2>What you get</h2>
        <p>
          Each extraction includes:
        </p>
        <ul className="list-disc list-inside space-y-1 text-[var(--text-body)]">
          <li>Span text and character offsets</li>
          <li>CUI and preferred term</li>
          <li>Semantic type (TUI) and group</li>
          <li>Similarity score (threshold at 0.55 by default)</li>
          <li>Assertion: present, absent, or possible</li>
          <li>Subject: patient, family, or other</li>
        </ul>

        <hr />

        <h2>Numbers</h2>
        <table>
          <tbody>
            <tr>
              <td>Model parameters</td>
              <td className="font-mono">9,864,198</td>
            </tr>
            <tr>
              <td>UMLS concepts indexed</td>
              <td className="font-mono">1,164,238</td>
            </tr>
            <tr>
              <td>Embedding dimensions</td>
              <td className="font-mono">768</td>
            </tr>
            <tr>
              <td>FAISS clusters</td>
              <td className="font-mono">2,048</td>
            </tr>
            <tr>
              <td>Throughput (RTX 3070)</td>
              <td className="font-mono">129 rows/sec</td>
            </tr>
          </tbody>
        </table>

        <hr />

        <h2>Limitations</h2>
        <p>
          No gold-standard labeled dataset was available. The evaluation numbers are behavior 
          metrics (row counts, score distributions), not accuracy metrics. Building a small 
          gold set is the next step.
        </p>
        <p>
          Common failure patterns:
        </p>
        <ul className="list-disc list-inside space-y-1 text-[var(--text-body)]">
          <li>Template leakage (section headers matching concepts)</li>
          <li>Abbreviation collisions ("BUN" has multiple meanings)</li>
          <li>Negation scope errors ("no HTN, DM, or CAD" → only first negated)</li>
        </ul>

        <hr />

        <h2>Try it</h2>
        <p>
          Head back to the <a href="/" className="text-[var(--text-primary)] underline">extractor</a> to 
          paste clinical text and see extractions in real time. Toggle between original spans 
          and preferred terminology to see the linking in action.
        </p>

        <hr />

        <div className="text-xs text-[var(--text-muted)] mt-8">
          <p className="mb-2">Citation:</p>
          <pre className="text-[10px]">
{`@misc{qassemf2026trmumls,
  author = {Qassemf, Layth M.},
  title = {trm-umls: a small embedding-distilled 
           umls concept linker for clinical text},
  year = {2026}
}`}
          </pre>
        </div>
      </article>
    </div>
  );
}
