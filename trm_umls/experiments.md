# experiments + decisions (trm-umls)

last updated: 2026-01-10

## tldr

- we are not pruning medgemma to “a few million params”. instead we use it (optionally) as a *teacher* to create labels.
- the runtime system is a small trm encoder (~10m params) + faiss over umls concept embeddings.
- “ensemble” in our work so far refers to **span proposal teachers** (ner models), not combining embedding vectors.
- embedding bakeoff (small): **sapbert(cls)** beat **bge-m3(dense)** on umls synonym→cui retrieval.

## the two “teacher” roles (important)

there are two separate places we use larger models:

1) span proposal teacher(s)
   - job: find plausible mention spans in raw notes.
   - output: `start/end/text` spans (no cuis).
   - current best: `d4data/biomedical-ner-all` + `nlpie/clinical-distilbert-i2b2-2010` + abbrev spans.

2) embedding teacher space (one model, one vector space)
   - job: turn text into vectors so we can retrieve the right cui.
   - used for:
     - building the umls concept index (faiss)
     - linking spans → cuis (silver labels)
     - training the small trm model (distillation)
   - current winner: sapbert cls pooling (see bakeoff).

we do **not** average/merge multiple embedding spaces at inference time. you pick one teacher space and everything must match it:

- concept vectors
- faiss index
- `embedding_metadata.json`
- the trm checkpoint (embedding dim + target space)

## full pipeline (end to end)

```mermaid
flowchart td
  note[note text] --> spans[span proposal teachers]
  spans --> link[link spans to cuis<br/>embed + faiss + rerank]
  link --> silver[silver span→cui pairs]
  silver --> mix[mix with base umls synonym pairs]
  mix --> train[train trm encoder<br/>to match teacher space]
  train --> ckpt[trm checkpoint]
  ckpt --> infer[inference: embed spans (trm) → faiss → cui]
```

## why this works (vs what failed before)

what we are doing now is narrow on purpose: “embed a span into a fixed teacher space, then nearest-neighbor lookup”.

common failure modes from earlier attempts and what changed:

- partial concept universe
  - if the correct cui is missing from the index, retrieval must pick a wrong cui.
  - fix: build the full umls concept index (same teacher space) before judging disambiguation quality.

- training on only note-derived spans
  - silver spans are noisy. if you train only on them, the embedding head can drift.
  - fix: always mix in clean umls synonym supervision (base pairs) so the model stays anchored.

- “semantic match” beats literal match too often
  - clinical notes have many short strings and abbreviations that collide.
  - fix: rerank within a small score band using lexical overlap + tui/group biases.

- severity/laterality/temporality are not encoded in cuis
  - umls often represents the base concept and not “mild/severe/left/right/acute”.
  - fix: extract these modifiers separately and strip them from the text used for embedding.

## repo state + cleanup (2026-01-10)

we removed older experiments and one-off artifacts so the repo stays readable and doesn’t retain sensitive clinical text.

local:
- no `archive/` folder
- `trm_umls/runs/` is treated as ephemeral (rerunnable) and kept empty by default

lambda (historical; instance may be terminated):
- `~/Medgemma/archive/20260110_071148/` (previous runs + notes subsets)
### what “scale up to full” means

most of the weird disambiguation errors are made worse when the faiss index only contains a small subset of cuis.

if the correct cui is not present in the index:

- linking (silver labels) must pick the “least-wrong” cui from what’s available
- inference must also pick the “least-wrong” cui

so the highest-impact “scale up” is usually:

1) build a larger/full umls concept index in the chosen teacher space
2) relink spans against that larger index
3) retrain trm on those cleaner silver labels

## what we ran on lambda (poc)

machine: `ubuntu@161.153.33.35`

### poc note sample

we used a small admission-note subset on lambda to iterate quickly (and to avoid copying the full 30k notes while we were still changing span filtering).

note: eval jsonl files include short context windows around spans, so treat `trm_umls/runs/*.jsonl` as sensitive and keep them private.

### span proposal (ensemble)

script:
- `trm_umls/scripts/teacher_label_notes_ensemble.py`

defaults:
- `d4data/biomedical-ner-all`
- `nlpie/clinical-distilbert-i2b2-2010`
- abbrev spans from `trm_umls/data/abbreviations.json`

key idea:
- we merge overlapping spans and keep teacher agreement metadata (`teacher_count`, `teachers`, `max_score`, …).

### linking spans → cuis (silver)

script:
- `trm_umls/scripts/link_spans_to_cuis.py`

inputs:
- spans jsonl from the ensemble teacher
- an embeddings dir with:
  - `umls_flat.index`
  - `embedding_metadata.json`
  - `umls_embeddings.npy`

filters we found useful for “clean enough” silver pairs:
- min embedding similarity (example: ~0.65)
- min lexical overlap (example: ~0.15)
- teacher agreement (`teacher_count >= 2`)

### training (distillation)

script:
- `trm_umls/train_multitask.py`

we trained in `--embedding-target concept` mode, meaning:
- the model learns to map text → the *concept (prefterm) embedding* for that cui
- we do **not** need synonym embeddings as targets

we also mixed in a large batch of “clean” umls synonym pairs (base supervision) so the model doesn’t learn only from noisy note spans.

### eval + review sets

script:
- `trm_umls/scripts/eval_notes_dir.py`

review helper:
- `trm_umls/scripts/make_review_set.py`

## key results so far

### embedding teacher bakeoff (umls synonym→cui retrieval)

run (historical; not kept after cleanup):
- `trm_umls/runs/teacher_bakeoff_small_20260110_075813.json`

sample:
- 20,000 concept prefterms + 20,000 synonym queries (restricted to those cuis)
- metric: retrieve correct cui from the concept subset

results:

| teacher | hit@1 | hit@5 | mrr |
|---|---:|---:|---:|
| sapbert(cls) | 0.9537 | 0.9751 | 0.9625 |
| bge-m3(dense) | 0.8853 | 0.9259 | 0.9016 |

interpretation:
- for umls-style synonym matching, sapbert is a better teacher space than bge-m3 (dense) in this setup.

### distillation signal (embedding validation metric)

from lambda checkpoints:
- baseline: `best_emb_mean_sim = 0.4983` (`trm_umls/checkpoints/baseline_epoch4/latest_state.json`)
- silver-trained (historical): `best_emb_mean_sim = 0.6103`
- full-index mixed training: `best_emb_mean_sim = 0.7062` (`trm_umls/checkpoints/latest_state.json`, epoch 10)

interpretation:
- the student moved meaningfully closer to the teacher space on held-out pairs.

### eval counts (not accuracy)

these are “how many extractions cleared threshold”, not “how correct are they”.

latest eval on 200 admission notes @ `threshold=0.55` with the poc index:
- baseline: 1,551 extracted mentions
- silver model: 6,610 extracted mentions

additional stats (from the same eval jsonl):

| run | notes with ≥1 extraction | mean score |
|---|---:|---:|
| baseline | 137/200 | 0.5740 |
| silver | 164/200 | 0.6102 |

interpretation:
- the silver-trained model tends to score more spans above threshold (more recall-ish), but it can include junk if span filtering is weak.

### full index eval (baseline vs new checkpoint)

using the full sapbert ivf index (1,164,238 concepts) on the 200-note admission poc set @ `threshold=0.55`:

| run | rows | notes with ≥1 extraction | mean score |
|---|---:|---:|---:|
| baseline (epoch 4, `trm_umls/checkpoints/baseline_epoch4/model.pt`) | 2,731 | 147/200 | 0.5773 |
| new (epoch 10, `trm_umls/checkpoints/model.pt`) | 10,331 | 181/200 | 0.6493 |

diff summary (span key = note_file + start/end):
- overlap spans: 2,118
- baseline-only spans: 613
- new-only spans: 8,213
- spans where the predicted cui changed: 1,466

lambda artifacts:
run artifacts (historical; not kept in-repo after cleanup):
- `trm_umls/runs/eval_baseline_epoch4_fullindex_candidates_t055_20260110_151521.jsonl`
- `trm_umls/runs/eval_new_epoch10_fullindex_candidates_t055_20260110_151521.jsonl`
- `trm_umls/runs/review_set_*_20260110_151521.jsonl`
- `trm_umls/runs/diff_baseline_epoch4_vs_new_epoch10_fullindex_t055_20260110_151521.jsonl` (no note text)

note: these eval files include span contexts and may contain sensitive text. keep private.

additional “sanity” metric (span text vs preferred term lexical overlap):
- baseline mean jaccard: 0.3145 (median: 0.25)
- new mean jaccard: 0.3986 (median: 0.3333)

## failure modes we saw (manual spot checks)

these are patterns, not patient-specific examples:

- templated headers/tables leaking into spans (vitals, lab tables, “planned procedures” blocks)
- “too-specific cui” wins when the correct generic cui is missing from the index
- abbreviations can collide (example pattern: `hgb` mapping to a non-hemoglobin concept if lexical tie-breaks are too weak)
- negation is still shallow (we improved `there is no …`, but the system is not a full negation engine)

we added conservative filters + better splitting in `trm_umls/pipeline.py` to reduce obvious template junk.

## what we did not finish yet

- mrrel relation rerank data (the pipeline supports it, but `mrrel_indptr.npy`/`mrrel_indices.npy` are not built on lambda right now).
- a true end-to-end clinical accuracy score (we have review-set tooling, but not a filled gold set).
- scaling span teachers beyond the small admission poc subset.

update (2026-01-10): we built a full sapbert concept index on lambda (1,164,238 vectors). for bulk linking speed we switched the default `umls_flat.index` symlink to point at an ivf index (`umls_ivf.index`) with `nlist=2048` and `nprobe=32`.

## next step (the “full scale” plan)

done (poc):
1) build a full sapbert concept embedding set + faiss index (same teacher space)
2) retrain trm against that full index (mixed training pairs)
3) rerun eval + generate review + diff files

next:
1) scale span proposals (500 notes → 5k+ notes) and rebuild silver pairs
2) rebuild `embeddings_train` with more silver pairs (same filters) and retrain
3) optionally add mrrel csr rerank data and compare on the fixed review set
4) fill a small gold set (50–200 rows) and track strict vs “related” cui accuracy

## reproducibility (paths + commands we used)

### bakeoff artifacts

lambda output:
- `~/Medgemma/trm_umls/runs/teacher_bakeoff_small_20260110_075813.json`

command (example):

```bash
python3 trm_umls/scripts/teacher_bakeoff.py \
  --base-embeddings-dir trm_umls/data/embeddings_sapbert_poc \
  --concept-count 20000 --query-count 20000 --k 5 \
  --run-sapbert --run-bge-m3 \
  --output trm_umls/runs/teacher_bakeoff_small_$(date +%Y%m%d_%H%M%S).json
```

### eval artifacts (poc index)

lambda outputs:
- `~/Medgemma/trm_umls/runs/eval_baseline_latest_t055_20260110_071838.jsonl`
- `~/Medgemma/trm_umls/runs/eval_silver_latest_t055_20260110_071838.jsonl`

note: these jsonl files include span contexts. keep them private.

### full index artifacts (sapbert)

lambda outputs:
- `~/Medgemma/trm_umls/data/embeddings_sapbert_full/umls_embeddings.npy` (1,164,238 × 768)
- `~/Medgemma/trm_umls/data/embeddings_sapbert_full/umls_ivf.index` (ivf; used for fast linking + retrieval)
- `~/Medgemma/trm_umls/data/embeddings_sapbert_full/umls_flat.index` (symlink → `umls_ivf.index`)

we keep the script defaults working by pointing `umls_flat.index` → `umls_ivf.index`.
