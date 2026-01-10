# trm-umls

tiny concept linker for clinical text:

- span → embedding (trm, ~10m params)
- embedding → cui (faiss nearest-neighbor over umls vectors)

the current “how-to” lives in `trm_umls/pipeline.md`.
the longer narrative writeup is `trm_umls/paper.md`.

## quick start

```bash
cd trm_umls
pip install -r requirements.txt
python3 pipeline.py --smoke
python3 pipeline.py --text "Patient denies HTN. Family history of DM."
```

## key scripts

- `trm_umls/scripts/teacher_label_notes_ensemble.py`: multi-teacher span proposals from notes
- `trm_umls/scripts/link_spans_to_cuis.py`: link spans → cuis in the teacher embedding space
- `trm_umls/scripts/build_silver_training_pairs.py`: mix base umls pairs + silver span pairs
- `trm_umls/train_multitask.py`: train the trm encoder
- `trm_umls/scripts/eval_notes_dir.py`: run extraction over a notes dir (jsonl)
- `trm_umls/scripts/make_review_set.py`: pick a small set of tricky rows for manual review
