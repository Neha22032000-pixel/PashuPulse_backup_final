# PashuPulse

PashuPulse is a low-connectivity rural livestock companion for cow, buffalo,
ox, and calves. It is an assistant, not a veterinarian: it gives safe
supportive care, routine husbandry help, myth correction, and escalation
guidance without prescribing medicines, injections, dosages, or procedures.

PashuPulse is intentionally framed as an **offline bounded livestock safety
assistant** for first-line guidance, myth correction, uncertainty-aware
follow-up, and escalation support. It is not an AI veterinarian or diagnosis
system.

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

## Offline Retrieval Cards

The offline retrieval-card pipeline builds a small runtime safety pack for
high-risk farmer questions. Cards are **not training rows**. They are grounded
in reviewed `source_claims.jsonl`; final-eval rubrics are used only for coverage
and tests; CPT chunks are not used as factual RAG grounding.

Build the current card pack:

```powershell
python pashu_saathi/scripts/build_retrieval_cards.py --out-dir pashu_saathi/data/processed/retrieval_cards
```

Generated artifacts:

- `retrieval_cards.jsonl`
- `retrieval_card_embeddings.npz`
- `retrieval_semantic_manifest.json`
- `retrieval_ablation_report.json`
- `retrieval_context_quality_report.json`
- `retrieval_demo_cases.jsonl`
- `retrieval_eval_queries.jsonl`
- `retrieval_card_manifest.json`
- `retrieval_card_quality_report.json`
- `retrieval_card_safety_report.json`

The retriever now supports three offline modes:

- `fallback`: BM25-compatible keyword scoring plus deterministic safety
  triggers.
- `phone_safe`: BM25 plus deterministic semantic vectors and safety triggers.
  This is the default phone-oriented mode.
- `demo_plus`: `phone_safe` plus optional top-5 reranking for hackathon demos.

The semantic artifact is intentionally small and auditable. It uses reviewed
card text plus a controlled livestock/myth alias ontology today, while the
manifest records future Android embedding-model candidates such as
`intfloat/multilingual-e5-small` and
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Safety-triggered
red cards are protected: semantic scoring or reranking can reorder candidates,
but cannot suppress hard safety matches.

Context Composer v2 is the runtime rendering layer. It stops dumping whole
cards into the prompt and instead renders only the most relevant facts, safe
actions, red flags, topic-specific avoid lines, and escalation text. Internal
hard bans remain in audit/validator metadata rather than becoming repetitive
visible boilerplate. The design principle is: **smallest useful safe answer**.

The composer also performs safety severity arbitration. Explicit protected
triggers control the final risk, but retrieved red cards do not create alarmist
tone unless the query or top card provides strong evidence. Lower-risk cards can
contribute facts, while the highest relevant risk controls escalation.

Protected triggers include oil/drench/puncture with bloat, hard pulling during
calving, dog-bite saliva exposure near children, abnormal milk sale pressure,
carcass handling after sudden death, photo-only diagnosis requests, and
medicine/injection/dose pressure.

If semantic artifacts fail on-device, the system falls back to BM25-style
keyword scoring plus deterministic safety routing. If retrieval confidence is
low, the assistant asks one critical follow-up or gives generic bounded holding
guidance. If generation fails validation, use the minimal safe fallback.

## Safety Boundaries

Hard bans include confident diagnosis, pills, injections, dosages, antibiotics,
painkillers, dewormers, vaccine administration, oil/drenching for bloat,
kerosene/alcohol, wound irritants, force-feeding, puncturing swelling, hard calf
pulling, fake contacts, unsafe milk/meat claims, and working injured oxen.

Runtime validators block medicine/dose/injection/procedure instructions,
unsupported diagnosis certainty, guaranteed milk/meat safety claims, and
irrelevant policy leakage.

## RAG + Gemma Inference Notebook

The Kaggle inference notebook/script lives in `kaggle_rag_gemma_inference/`.
It is intentionally inference-only, not an SFT run. It compares:

- Gemma alone.
- Gemma with Context Composer v2 retrieval context.
- Gemma with Context Composer v2 plus post-generation validator/fallback.

Run locally as a dry-run:

```powershell
$env:PASHU_RAG_DRY_RUN = "1"
$env:PASHU_RAG_OUT_DIR = "test_runs/rag_gemma_inference_dry"
python kaggle_rag_gemma_inference/rag_gemma_inference.py
```

Expected Kaggle outputs:

- `rag_generation_trace.jsonl`
- `rag_generation_predictions.jsonl`
- `rag_validator_report.json`
- `rag_demo_table.csv`
- `rag_notebook_manifest.json`

The notebook records retrieved card IDs, rendered context, suppressed fields,
raw Gemma answer, validator result, final answer, and fallback use for every
row. Use this before more SFT to decide whether failures are prompt/composer,
validator, retrieval, or model-style problems.

## Out Of Scope

- Disease diagnosis.
- Medicine prescription or dosage.
- Injection or vaccine administration guidance.
- Surgery or invasive procedure instructions.
- Replacement for veterinarians or emergency care.
- Guaranteed milk/meat safety.
- Image-only disease certainty.
- Broad livestock medical reasoning beyond first-line safety guidance.

## Kaggle Account Isolation

This track may use the teammate Kaggle credential folder only inline:

```powershell
$env:KAGGLE_CONFIG_DIR = "C:\Users\risha\Documents\New project 5\.kaggle_2"
```

Do not set that globally. Kaggle runs for this project must force T4 and never
use P100.
