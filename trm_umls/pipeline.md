# trm-umls pipeline (concept extraction)

## goal

take clinical text and return linked umls concepts (cui + preferred term + tui), fast enough to feel like a dictionary lookup, but with better synonym/semantic coverage than pure string matching.

the runtime model is a small trm encoder (~10m params) that embeds spans into the same vector space as a stronger “teacher” embedding model (currently sapbert-style 768d). retrieval is a faiss nearest-neighbor search over precomputed umls concept vectors.

## what runs at inference time

1. split note text into candidate spans (with simple header/templating cleanup)
2. expand common abbreviations (htn → hypertension, etc.)
3. embed each span with the trm model
4. faiss search → top-k candidate cuis
5. rerank within a small score band (lexical overlap + optional clinical biases + optional mrrel relation signal)
6. emit cui + preferred term + tui (+ optional assertion/subject labels)

entrypoints:

- python api: `trm_umls/pipeline.py` (`TRMUMLSPipeline.extract()`)
- cli: `python3 trm_umls/pipeline.py --text "..."`

## repo map (what each file is)

core:

- `trm_umls/pipeline.py`: inference pipeline (span cleanup + abbrev expansion + embed + faiss + rerank)
- `trm_umls/models/trm_text_encoder.py`: the trm encoder + heads (assertion/subject)
- `trm_umls/models/trm_core.py`: core trm blocks + config
- `trm_umls/train_multitask.py`: training loop (embedding distillation + optional heads)
- `trm_umls/utils/abbreviations.py`: abbreviation expander

docs:

- `trm_umls/pipeline.md`: how the pipeline works
- `trm_umls/experiments.md`: what we tried + results + decisions

scripts (offline / one-time jobs):

- `trm_umls/scripts/extract_umls.py`: parse umls exports into json files used by the pipeline
- `trm_umls/scripts/generate_embeddings_hf_local.py`: build teacher embeddings (`umls_embeddings.npy`, `embedding_metadata.json`, etc.)
- `trm_umls/scripts/build_faiss_index.py`: build a faiss index over `umls_embeddings.npy`
- `trm_umls/scripts/extract_mrrel_subset.py`: build the optional mrrel csr graph for relation rerank

scripts (silver data + eval):

- `trm_umls/scripts/teacher_label_notes_ensemble.py`: propose mention spans from notes (multi-teacher ner)
- `trm_umls/scripts/link_spans_to_cuis.py`: link spans → cuis (teacher embeddings + faiss + rerank)
- `trm_umls/scripts/build_silver_training_pairs.py`: mix base umls supervision + silver span pairs
- `trm_umls/scripts/eval_notes_dir.py`: run extraction over a notes dir (jsonl)
- `trm_umls/scripts/make_review_set.py`: sample a small set of rows for manual review

## offline assets (build once, reuse many times)

### 1) umls concept store

minimum files:

- `trm_umls/data/umls/tui_mappings.json` (cui → tui list)
- `trm_umls/data/abbreviations.json` (optional, for expansion)

optional:

- `trm_umls/data/umls/mrrel_indptr.npy` + `mrrel_indices.npy` (csr graph for relation rerank)

### 2) embedding index (teacher space)

minimum files in an “embeddings dir” (example: `trm_umls/data/embeddings_sapbert_poc/`):

- `umls_flat.index` (faiss)
- `embedding_metadata.json` (parallel arrays: cuis, prefterms, embedding_dim, …)
- `umls_embeddings.npy` (float32 vectors aligned to the faiss rows)

note: the current lambda setup uses a 200k-row poc index for speed. for real coverage you want the full index (1m+ cuis).

## training data (how we get better labels without manual annotation)

we generate “silver” span→cui pairs from notes and mix them with clean umls synonym pairs.

### step a: span proposals (teachers)

`trm_umls/scripts/teacher_label_notes_ensemble.py`

default teachers:

- `d4data/biomedical-ner-all` (token-classification ner)
- `nlpie/clinical-distilbert-i2b2-2010` (token-classification ner)
- optional abbreviation spans from `trm_umls/data/abbreviations.json`

output: per-note jsonl with merged spans + teacher agreement metadata.

### step b: link spans to cuis (silver labels)

`trm_umls/scripts/link_spans_to_cuis.py`

for each span:

- embed the span text in the teacher space
- faiss retrieve candidate cuis
- filter by min similarity + min lexical overlap + (optional) tui/group filters

output: jsonl with span + chosen cui (+ candidates/metadata for debugging).

### step c: build a mixed training set

`trm_umls/scripts/build_silver_training_pairs.py`

creates a new “embedding dir” that contains:

- sampled base umls synonym pairs (stable supervision)
- additional silver span pairs (note-derived supervision)

this avoids training only on noisy spans, which tends to hurt embedding quality.

### step d: train trm

`trm_umls/train_multitask.py`

key idea:

- the trm embedding head learns to match the teacher vector space
- optional classification heads can learn assertion/subject labels if you provide datasets

## evaluation + manual review loop

1. run eval on a notes directory:

```bash
python3 trm_umls/scripts/eval_notes_dir.py \
  --notes-dir <dir_of_txt_notes> \
  --output-jsonl trm_umls/runs/eval_<tag>.jsonl \
  --checkpoint <model.pt> \
  --index-path <embeddings_dir>/umls_flat.index \
  --metadata-path <embeddings_dir>/embedding_metadata.json \
  --tui-mappings-path trm_umls/data/umls/tui_mappings.json \
  --threshold 0.55 \
  --clinical-rerank \
  --include-candidates \
  --dedupe
```

1b. export the eval jsonl to a color-coded xlsx:

```bash
python3 trm_umls/scripts/export_eval_jsonl_to_xlsx.py \
  --eval-jsonl trm_umls/runs/eval_<tag>.jsonl \
  --output-xlsx trm_umls/runs/eval_<tag>.xlsx
```

2. create a small review set:

```bash
python3 trm_umls/scripts/make_review_set.py \
  --eval-jsonl trm_umls/runs/eval_<tag>.jsonl \
  --output-jsonl trm_umls/runs/review_set_<tag>.jsonl \
  --n 50
```

3. (optional) fill `label.gold_cui` and score:

```bash
python3 trm_umls/scripts/score_review_set.py --review-jsonl trm_umls/runs/review_set_<tag>.jsonl
```

## what to improve next (highest impact)

1. use a full umls index on the same teacher space you train against (poc indexes skew results)
2. scale silver labeling from 500 notes → 5k+ and loosen filters to get more “clean enough” pairs
3. tighten span filtering for templated note sections (vitals, lab tables, admin text)
4. add a small “reject span” model or stronger heuristics if junk spans dominate
5. if disambiguation is still weak, enable relation rerank (mrrel) and/or add a second teacher for hard cases
