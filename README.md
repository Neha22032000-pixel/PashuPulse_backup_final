# PashuPulse

PashuPulse is a low-connectivity rural livestock companion for cow, buffalo,
ox, and calves. It is an assistant, not a veterinarian: it gives safe
supportive care, routine husbandry help, myth correction, and escalation
guidance without prescribing medicines, injections, dosages, or procedures.

## What Is Here

- auditable seed-bank and expansion tooling
- cleaned SFT package builders and Kaggle T4 runners
- final-eval local LLM judge workflow
- broad CPT/DAPT source-corpus candidate tooling
- tests for dataset gates, SFT contamination checks, and judge/eval plumbing

The current champion adapter remains `checkpoint-250` from the cleaned full SFT
run. Later continuation runs are treated as experiments unless they beat the
champion in sealed final eval plus manual review.

## Dataset And Training Flow

```powershell
python pashu_saathi/scripts/generate_dataset.py --phase full_expansion_candidate --out-dir pashu_saathi/data/processed/full_expansion
python pashu_saathi/scripts/validate_dataset.py pashu_saathi/data/processed/full_expansion --phase full_expansion_candidate
python pashu_saathi/scripts/prepare_kaggle_sft_package.py --mode full
python -m pytest pashu_saathi/tests -c pashu_saathi/pyproject.toml -p no:cacheprovider
```

SFT exports stay gated. Final eval, adversarial prompts, judge labels,
predictions, and `EVAL_ONLY_DO_NOT_TRAIN` rows must never enter training
packages.

## Broad CPT Corpus

The CPT/DAPT pipeline is intentionally broader than the SFT/RAG source registry.
Its goal is domain familiarity: livestock vocabulary, husbandry concepts,
symptom descriptions, dairy workflows, and extension style. International
official, university, extension, NGO, and reputable educational sources are
allowed.

Build the current candidate corpus:

```powershell
python pashu_saathi/scripts/build_cpt_corpus.py --out-dir pashu_saathi/data/processed/cpt_corpus
```

Generated artifacts:

- `candidate_cpt_sources.jsonl`
- `accepted_cpt_sources.jsonl`
- `quarantined_cpt_sources.jsonl`
- `rejected_cpt_sources.jsonl`
- `cpt_clean_chunks.jsonl`
- `cpt_corpus_manifest.json`
- `cpt_source_quality_report.json`
- `cpt_safety_filter_report.json`

CPT accepts broad educational material but rejects or quarantines content that
would teach bad priors: miracle cures, unsafe folk remedies, raw drug doses,
injection routes, procedure walkthroughs, anti-vaccine claims, product spam, and
bad OCR. Quarantined CPT chunks are not allowed as SFT/RAG factual grounding.

## CPT/DAPT Research Artifacts

The CPT/DAPT research pass is separate from training. It asks whether continued
pretraining is worth attempting for livestock knowledge grounding before any
parameter or Kaggle launch discussion:

```powershell
python pashu_saathi/scripts/build_cpt_research_artifacts.py --out-dir pashu_saathi/data/processed/cpt_research --library-dir pashu_saathi/data/sources/offline_library_v1
```

Generated artifacts include source-family train/dev/test splits, a corpus audit,
an eval-only source-derived question bank, an experiment matrix, and a go/no-go
decision template. These artifacts explicitly keep `training_allowed` and
`sft_allowed` false.

Research docs:

- `docs/cpt_dapt_research_brief.md`
- `docs/cpt_eval_design.md`

## Offline Source Library

The broader offline source acquisition workflow downloads raw sources, extracts
text, builds cleaned chunks, and records coverage/safety manifests for a later
grounding-selection pass:

```powershell
python pashu_saathi/scripts/build_offline_source_library.py --out-dir pashu_saathi/data/sources/offline_library_v1
```

This library is coverage-first. It preserves English and Hindi/Hinglish-relevant
sources for cow, buffalo, ox/bullock, and calf topics, but it is not directly
approved for SFT or farmer-facing retrieval until a separate grounding review
chooses the safe subset.

Current library snapshot:

- `74` cataloged source records
- `60` accepted or accepted-after-stripping sources
- `64` raw downloaded files preserved offline
- `3,799` clean chunks
- about `2.54M` clean tokens
- coverage gate passes for cow, buffalo, calf, ox/bullock, feeding, water,
  shed hygiene, milk hygiene, heat/cold stress, calf care, pregnancy/calving,
  bloat, wounds, diarrhea, FMD-like signs, poisoning/spoiled feed, parasites,
  bites, and working ox care

The raw, extracted, and cleaned source files are stored with Git LFS because the
offline corpus is large. After cloning, run:

```powershell
git lfs pull
```

Useful teammate starting points:

- `data/sources/offline_library_v1/manifests/source_download_manifest.jsonl`
- `data/sources/offline_library_v1/manifests/source_quality_manifest.jsonl`
- `data/sources/offline_library_v1/reports/source_coverage_report.json`
- `data/sources/offline_library_v1/reports/source_safety_filter_report.json`
- `data/sources/offline_library_v1/cpt_clean_chunks.jsonl`

The next pass should select a smaller strict grounding subset from the accepted
sources. Do not treat the broad CPT/offline library as final answer authority.

## Safety Boundaries

Hard bans include confident diagnosis, pills, injections, dosages, antibiotics,
painkillers, dewormers, vaccine administration, oil/drenching for bloat,
kerosene/alcohol, wound irritants, force-feeding, puncturing swelling, hard calf
pulling, fake contacts, unsafe milk/meat claims, and working injured oxen.

## Kaggle Account Isolation

This track may use the teammate Kaggle credential folder only inline:

```powershell
$env:KAGGLE_CONFIG_DIR = "C:\Users\risha\Documents\New project 5\.kaggle_2"
```

Do not set that globally. Kaggle runs for this project must force T4 and never
use P100.
